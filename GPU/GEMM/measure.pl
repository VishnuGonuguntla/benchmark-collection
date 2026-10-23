#!/usr/bin/env perl
use strict;
use warnings;
use Time::HiRes qw(sleep time);
use POSIX qw(strftime :sys_wait_h);
use Data::Dumper;

# defaults
my $interval    = 10;
my $cpu_mode    = "off";           # intel|amd|off
my $cpu_csv     = "cpu-measurement.csv";
my $gpu_mode    = "off";             # nvidia|amd|off
my $gpu_csv     = "gpu-measurement.csv";
my $gpu_device  = undef;   # optional GPU device ID
my $write_header = 0;      # default to OFF

my $cpu_fh;  # declare globally for this script

my @cmd_args;

# parse args
while (@ARGV) {
    my $arg = shift @ARGV;
	
    if ($arg eq '--cpu-csv') {
        $cpu_csv = shift(@ARGV) or die "Missing filename after --cpu-csv\n";
    } elsif ($arg eq '--gpu-csv') {
        $gpu_csv = shift(@ARGV) or die "Missing filename after --gpu-csv\n";
    } elsif ($arg eq '--interval') {
        $interval = shift(@ARGV) or die "Missing number after --interval\n";
        die "Interval must be positive integer\n" unless $interval =~ /^\d+$/ && $interval > 0;
    } elsif ($arg eq '--cpu') {
        $cpu_mode = shift(@ARGV) // die "Missing value after --cpu\n";
        die "Invalid --cpu option\n" unless $cpu_mode =~ /^(intel|amd|off)$/;
    } elsif ($arg eq '--gpu') {
        $gpu_mode = shift(@ARGV) // die "Missing value after --gpu\n";
        die "Invalid --gpu option\n" unless $gpu_mode =~ /^(nvidia|amd|off)$/;
    } elsif ($arg eq '--gpu-device') {
        $gpu_device = shift(@ARGV) // die "Missing ID after --gpu-device\n";
        die "Invalid GPU device ID\n" unless $gpu_device =~ /^\d+$/;
	} elsif ($arg eq '--header') {
        $write_header = 1;
    } else {
        push @cmd_args, $arg, @ARGV;
        last;
    }
}

my $cmd = join(" ", @cmd_args);

# require a workload
if (!$cmd) {
    die "This script requires a workload command to run. Usage: $0 [options] <your command/executable>\n";
}

# if both off, exit
if ($cpu_mode eq 'off' && $gpu_mode eq 'off') {
    die "Both CPU and GPU monitoring are off. Nothing to do. Exiting.\n";
}

# paths
my $cpu_path   = "/sys/devices/system/cpu";
my @rapl_paths = </sys/class/powercap/intel-rapl/intel-rapl:*>;
my %last_energy = ();

# detect cores
opendir(my $dh, $cpu_path) or die "Cannot open $cpu_path: $!";
my @cpus = sort grep { /^cpu\d+$/ && -d "$cpu_path/$_" } readdir($dh);
closedir $dh;

# prepare CPU CSV header depending on cpu_mode
my @cpu_headers = ("Time");

my @temp_files = ();
if ($cpu_mode eq 'intel') {
    my @hwmons = ("/sys/class/hwmon/hwmon1", "/sys/class/hwmon/hwmon2");
    for my $hwm (@hwmons) {
        next unless -d $hwm;
        opendir(my $tdh, $hwm) or next;
        my @temps = grep { /^temp\d+_input$/ } readdir($tdh);
        closedir $tdh;
        for my $tf (sort { ($a =~ /temp(\d+)_input/)[0] <=> ($b =~ /temp(\d+)_input/)[0] } @temps) {
            next if $tf =~ /^temp0_input$/;
            push @temp_files, "$hwm/$tf";
        }
    }
    push @cpu_headers, map { "CoreTemp$_" } (0 .. $#temp_files) if @temp_files;
} elsif ($cpu_mode eq 'amd') {
    # get all hwmon folders
    my @all_hwmon = sort { 
		($a =~ /hwmon(\d+)/)[0] <=> ($b =~ /hwmon(\d+)/)[0] 
	} grep { -d $_ } </sys/class/hwmon/hwmon*>;
	# pick the last two folders
    my @hws = @all_hwmon[-3 .. -1];

    my $counter = 0;
    for my $hwm (@hws) {
        next unless -d $hwm;
        for my $t (3..10) {
            my $file = "$hwm/temp${t}_input";
            if (-e $file) {
                push @temp_files, $file;
                push @cpu_headers, "CCDTemp" . $counter;
            }
            $counter++;
        }
    }
}

# frequencies (same path for both)
push @cpu_headers, map { "CoreFreq" . substr($_,3) } @cpus;

# rapl packages
for my $pkg (@rapl_paths) {
    my $file = "$pkg/energy_uj";
    if (-e $file) {
        my $val = read_file($file);
        $last_energy{$pkg} = $val if defined $val;
    }
}
push @cpu_headers, map { "PackagePower" . substr($_, rindex($_,":")+1) } @rapl_paths;

if ($cpu_mode ne 'off') {
    # open CPU CSV
    my $need_cpu_header = (! -e $cpu_csv && $write_header);
    open($cpu_fh, ">>", $cpu_csv) or die "Cannot write to $cpu_csv: $!";
    $cpu_fh->autoflush(1);
    print $cpu_fh join(",", @cpu_headers), "\n" if $need_cpu_header;
}

# --- Workload Execution ---
my $work_pid = fork();
die "Cannot fork: $!" unless defined $work_pid;
if ($work_pid == 0) {
    exec($cmd) or die "Failed to exec workload: $!";
}

print "Launched workload (pid $work_pid). Monitoring interval: $interval s\n";

$SIG{INT} = sub { terminate_and_cleanup(); exit 1; };
$SIG{TERM} = sub { terminate_and_cleanup(); exit 1; };

my $elapsed = 0;
while (1) {
    my $wp = waitpid($work_pid, WNOHANG);
    last if $wp == $work_pid;

    my $timestamp = $elapsed;
    my $start = time();

    if ($cpu_mode ne 'off') {
        my @temps = ();
        if ($cpu_mode eq 'intel') {
            @temps = map { my $v = read_file($_); defined $v ? $v/1000 : "" } @temp_files;
        } elsif ($cpu_mode eq 'amd') {
            foreach my $tf (@temp_files) {
                my $v = read_file($tf);
                push @temps, defined $v ? $v/1000 : "";
            }
        }

		my @freqs = ();
        for my $cpu (@cpus) {
            my $f = "$cpu_path/$cpu/cpufreq/scaling_cur_freq";
            my $val = read_file($f);
            push @freqs, defined $val ? $val/1000 : "";
        }

        my @powers = ();
        for my $pkg (@rapl_paths) {
            my $file = "$pkg/energy_uj";
            my $val = read_file($file);
            if (defined $val) {
                my $prev = $last_energy{$pkg} || $val;
                my $delta = $val - $prev;
                $delta += 2**32 if $delta < 0;
                my $power = $delta / ($interval * 1e6);
                push @powers, sprintf("%.3f", $power > 0 ? $power : 0);
                $last_energy{$pkg} = $val;
            } else {
				push @powers, "";
			}
        }
        print $cpu_fh join(",", $timestamp, @temps, @freqs, @powers), "\n";
    }

    # --- GPU Collection (Multi-GPU Enabled) ---
    # --- GPU Collection (Iterative Index Method) ---
    if ($gpu_mode eq 'nvidia') {
        # 1. Determine which GPU IDs to iterate over
        my @target_gpus;
        if (defined $gpu_device) {
            push @target_gpus, $gpu_device;
        } else {
            # Get the total count of GPUs if no specific device is set
            my $count_raw = `nvidia-smi --query-gpu=count --format=csv,noheader,nounits`;
            my ($count) = $count_raw =~ /(\d+)/;
            @target_gpus = (0 .. ($count - 1)) if $count;
        }

        # 2. Iterate through each GPU and collect data
        if (open(my $gfh, ">>", $gpu_csv)) {
            $gfh->autoflush(1);
            if (-s $gpu_csv == 0 && $write_header) {
                print $gfh "GPU_Idx,Pwr_W,Temp_C,Temp_M,Sm_Pct,Mem_Pct,Enc_Pct,Dec_Pct,MClk_MHz,PClk_MHz,MemPwr_W\n";
            }

            foreach my $idx (@target_gpus) {
                # A. Get standard metrics for THIS index
                my $q_fields = "power.draw,temperature.gpu,temperature.memory,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,clocks.current.memory,clocks.current.graphics";                my $std_data = `nvidia-smi -i $idx --query-gpu=$q_fields --format=csv,noheader,nounits 2>/dev/null`;
                chomp($std_data) if $std_data;
                $std_data =~ s/\s//g; # Clean formatting

                # B. Get Memory Power for THIS index using your grep logic
                my $mem_power_raw = `nvidia-smi -i $idx -q 2>/dev/null | grep -A 4 "GPU Memory Power Readings" | grep "Power Draw" | head -n 1`;
                my $m_pwr = "0.00";
                if ($mem_power_raw && $mem_power_raw =~ /:\s*([\d\.]+)\s*W/) {
                    $m_pwr = $1;
                }

                # C. Write the combined line
                if ($std_data) {
                    print $gfh "$idx,$std_data,$m_pwr\n";
                }
            }
            close $gfh;
        }
    }

    # --- GPU Collection (AMD / ROCm) ---
    # Output column order: idx,pwr,temp_gpu,temp_mem,util_gpu,util_mem,util_enc,util_dec,clk_mem,clk_gfx,mem_pwr,clk_fclk,clk_socclk
    if ($gpu_mode eq 'amd') {
        # 1. Determine which GPU IDs to query
        my @target_gpus;
        if (defined $gpu_device) {
            push @target_gpus, $gpu_device;
        } else {
            my $id_raw = `rocm-smi --showid 2>/dev/null`;
            my %seen;
            my @found = sort { $a <=> $b } grep { !$seen{$_}++ } ($id_raw =~ /GPU\[(\d+)\]/g);
            @target_gpus = @found ? @found : (0);
        }

        # 2. Iterate through each GPU and collect data
        if (open(my $gfh, ">>", $gpu_csv)) {
            $gfh->autoflush(1);
            if (-s $gpu_csv == 0 && $write_header) {
                print $gfh "GPU_Idx,Pwr_W,Temp_C,Temp_M,Sm_Pct,Mem_Pct,Enc_Pct,Dec_Pct,MClk_MHz,PClk_MHz,MemPwr_W,FClk_MHz,SOCClk_MHz\n";
            }
            foreach my $idx (@target_gpus) {
                my $raw = `rocm-smi -d $idx --showpower --showtemp --showuse --showmemuse --showclocks 2>/dev/null`;

                # Power draw (W) - Targeted specifically at Graphics Package Power
                my $pwr = "0.00";
                # This regex looks specifically for "Graphics Package Power" and ignores "Socket" if present elsewhere
                if ($raw =~ /Graphics Package Power.*?:\s*([\d\.]+)/i) {
                    $pwr = $1;
                }

                # GPU die (junction) temperature (C)
                my $temp_gpu = "";
                if ($raw =~ /Temperature \(Sensor junction\).*?:\s*([\d\.]+)/i) {
                    $temp_gpu = $1;
                }

                # Memory temperature (C)
                my $temp_mem = "";
                if ($raw =~ /Temperature \(Sensor memory\).*?:\s*([\d\.]+)/i) {
                    $temp_mem = $1;
                }

                # GPU utilization (%)
                my $util_gpu = "";
                if ($raw =~ /GPU use \(%\).*?:\s*([\d\.]+)/i) {
                    $util_gpu = $1;
                }

                # Memory utilization (%) - Uses VRAM allocation percentage
                my $util_mem = "";
                if ($raw =~ /GPU Memory Read\/Write Activity \(%\).*?:\s*([\d\.]+)/i) {
                    $util_mem = $1;
                }

                # Encoder/Decoder (ROCm internal metrics vary, placeholders used)
                my ($util_enc, $util_dec) = ("", "");

                # GFX clock (MHz)
                my $clk_gfx = "";
                if	($raw =~ /sclk clock level[^:]*:[^(]*\((\d+)Mhz\)/i)		 { $clk_gfx = $1 }
                elsif ($raw =~ /GFX Clock[^s][^:]*:\s*(\d+)/i)					 { $clk_gfx = $1 }
                elsif ($raw =~ /GFX Clocks(?:[^\n]*\n)+?[^\n]*?(\d+)Mhz\s*\*/si)  { $clk_gfx = $1 }

                # MEM clock (MHz)
                my $clk_mem = "";
                if	($raw =~ /mclk clock level[^:]*:[^(]*\((\d+)Mhz\)/i)		 { $clk_mem = $1 }
                elsif ($raw =~ /MEM Clock[^s][^:]*:\s*(\d+)/i)					 { $clk_mem = $1 }
                elsif ($raw =~ /MEM Clocks(?:[^\n]*\n)+?[^\n]*?(\d+)Mhz\s*\*/si)  { $clk_mem = $1 }

                # Memory/HBM subsystem power (W)
                my $mem_pwr = "0.00";
                if ($raw =~ /(?:HBM|Memory)\s+Power.*?:\s*([\d\.]+)/i) {
                    $mem_pwr = $1;
                }

                # FCLK clock (MHz)
                my $clk_fclk = "";
                if ($raw =~ /fclk clock level[^:]*:[^(]*\((\d+)Mhz\)/i) { $clk_fclk = $1 }

                # SOCCLK clock (MHz)
                my $clk_socclk = "";
                if ($raw =~ /socclk clock level[^:]*:[^(]*\((\d+)Mhz\)/i) { $clk_socclk = $1 }

                # Print without timestamp
                print $gfh "$idx,$pwr,$temp_gpu,$temp_mem,$util_gpu,$util_mem,$util_enc,$util_dec,$clk_mem,$clk_gfx,$mem_pwr,$clk_fclk,$clk_socclk\n";
            }
            close $gfh;
        }
    }

    my $elapsed_collect = time() - $start;
    my $sleep_time = $interval - $elapsed_collect;
    sleep($sleep_time > 0 ? $sleep_time : 0);
    $elapsed += $interval;
}

print "Workload finished, stopping measurements.\n";
terminate_and_cleanup();
exit 0;

sub terminate_and_cleanup {
    if ($work_pid) {
        my $wp = waitpid($work_pid, WNOHANG);
        if ($wp != $work_pid) {
            kill 'TERM', $work_pid;
            waitpid($work_pid, 0);
            print "Stopped workload (pid $work_pid)\n";
        }
    }
    close $cpu_fh if $cpu_fh;
}

sub read_file {
    my ($file) = @_;
    return undef unless defined $file && -e $file;
    open my $fh, '<', $file or return undef;
    my $val = <$fh>;
    close $fh;
    chomp($val) if defined $val;
    return $val;
}