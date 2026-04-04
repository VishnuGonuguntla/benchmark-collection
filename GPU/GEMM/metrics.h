#ifndef METRICS_H
#define METRICS_H

// GPU telemetry monitoring — NVML (CUDA) or ROCm SMI (HIP).
// Everything lives inside #ifdef METRICS so the header is always safe to
// include but compiles to nothing when metrics are disabled.

#ifdef METRICS

#include "portability.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>

// Averaged metric snapshot returned by GpuMonitor::averages().
struct MetricsAvg {
  double power;    // W
  double clock;    // MHz
  double temp;     // C
  double gpu_util; // %
  double mem_util; // %
};

// Thin wrapper around NVML / ROCm SMI that samples metrics on a background
// thread at a fixed interval (~50 ms).  Usage:
//
//   GpuMonitor mon;
//   if (!mon.init(device_id, gpu_id)) return 1;
//   mon.start();          // resets accumulators, spawns thread
//   ... run kernels ...
//   mon.stop();           // joins thread
//   MetricsAvg a = mon.averages();
//   mon.shutdown();       // tear down backend
//
// start()/stop() may be called repeatedly (e.g. per sweep size).
struct GpuMonitor {
  // Accumulators (written only by the sampling thread)
  double totalPower;
  double totalClock;
  double totalTemp;
  unsigned long long totalGpuUtil;
  unsigned long long totalMemUtil;
  int samples;

  // Thread control
  std::atomic<bool> _active;
  std::thread _worker;

  // Backend handle
#ifndef _HIP
  nvmlDevice_t _nvml_dev;
#else
  uint32_t _rsmi_idx;
#endif

  // Initialize the monitoring backend.  Returns true on success.
  inline bool init(int device_id, int gpu_id)
  {
#ifndef _HIP
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to init NVML: " << nvmlErrorString(r) << std::endl;
      return false;
    }
    r = nvmlDeviceGetHandleByIndex(device_id, &_nvml_dev);
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to get NVML device: " << nvmlErrorString(r) << std::endl;
      return false;
    }
    (void)gpu_id;
#else
    rsmi_status_t r = rsmi_init(0);
    if (r != RSMI_STATUS_SUCCESS) {
      const char *err_str;
      rsmi_status_string(r, &err_str);
      std::cerr << "Failed to init ROCm SMI: " << err_str << std::endl;
      return false;
    }
    // gpu_id matches ROCm SMI device ordering (both use BDF/PCI bus order)
    _rsmi_idx = (uint32_t)gpu_id;
    (void)device_id;
#endif
    return true;
  }

  // Reset accumulators and start the sampling thread.
  inline void start()
  {
    totalPower   = 0.0;
    totalClock   = 0.0;
    totalTemp    = 0.0;
    totalGpuUtil = 0;
    totalMemUtil = 0;
    samples      = 0;
    _active.store(true);

    _worker = std::thread([this]() {
      while (_active.load()) {
#ifndef _HIP // NVML
        {
          unsigned int power_mW, clock_MHz, temp_C;
          nvmlUtilization_t util;
          if (nvmlDeviceGetPowerUsage(_nvml_dev, &power_mW) == NVML_SUCCESS)
            totalPower += power_mW / 1000.0;
          if (nvmlDeviceGetClockInfo(_nvml_dev, NVML_CLOCK_GRAPHICS, &clock_MHz) == NVML_SUCCESS)
            totalClock += clock_MHz;
          if (nvmlDeviceGetTemperature(_nvml_dev, NVML_TEMPERATURE_GPU, &temp_C) == NVML_SUCCESS)
            totalTemp += temp_C;
          if (nvmlDeviceGetUtilizationRates(_nvml_dev, &util) == NVML_SUCCESS) {
            totalGpuUtil += util.gpu;
            totalMemUtil += util.memory;
          }
        }
#else // ROCm SMI
        {
          uint64_t power_uW;
          RSMI_POWER_TYPE power_type;
          rsmi_frequencies_t freqs;
          int64_t  temp_mC;
          uint32_t busy_pct;
          if (rsmi_dev_power_get(_rsmi_idx, &power_uW, &power_type) == RSMI_STATUS_SUCCESS)
            totalPower += (double)power_uW / 1e6;
          if (rsmi_dev_gpu_clk_freq_get(_rsmi_idx, RSMI_CLK_TYPE_SYS, &freqs) == RSMI_STATUS_SUCCESS)
            totalClock += (double)freqs.frequency[freqs.current] / 1e6;
          if (rsmi_dev_temp_metric_get(_rsmi_idx, RSMI_TEMP_TYPE_JUNCTION,
                                       RSMI_TEMP_CURRENT, &temp_mC) == RSMI_STATUS_SUCCESS)
            totalTemp += temp_mC / 1000.0;
          if (rsmi_dev_busy_percent_get(_rsmi_idx, &busy_pct) == RSMI_STATUS_SUCCESS)
            totalGpuUtil += busy_pct;
          if (rsmi_dev_memory_busy_percent_get(_rsmi_idx, &busy_pct) == RSMI_STATUS_SUCCESS)
            totalMemUtil += busy_pct;
        }
#endif
        samples++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  // Stop the sampling thread.
  inline void stop()
  {
    _active.store(false);
    if (_worker.joinable())
      _worker.join();
  }

  // Compute averaged metrics from accumulated samples.
  inline MetricsAvg averages() const
  {
    MetricsAvg avg = {};
    if (samples > 0) {
      avg.power    = totalPower   / samples;
      avg.clock    = totalClock   / samples;
      avg.temp     = totalTemp    / samples;
      avg.gpu_util = (double)totalGpuUtil / samples;
      avg.mem_util = (double)totalMemUtil / samples;
    }
    return avg;
  }

  // "SM Util" (NVIDIA) or "CU Util" (AMD)
  static inline const char *util_label()
  {
#ifndef _HIP
    return "SM Util";
#else
    return "CU Util";
#endif
  }

  // Shut down the monitoring backend.
  inline void shutdown()
  {
#ifndef _HIP
    nvmlShutdown();
#else
    rsmi_shut_down();
#endif
  }
};

#endif // METRICS
#endif // METRICS_H
