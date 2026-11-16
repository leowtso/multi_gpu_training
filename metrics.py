"""Throughput/utilisation measurement helpers."""

import threading

import torch


class GpuUtilSampler:
    """Samples NVML 'GPU utilisation' in a background thread.

    NVML utilisation = % of time *any* kernel was running on the GPU. It says
    nothing about how efficiently the kernel used the SMs, so a GPU can show
    100% here at 20% MFU. Report it alongside MFU, never instead of it.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.5):
        self.device, self.interval = device, interval_s
        self.samples = []
        self._active = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        if device.type == "cuda":
            try:
                torch.cuda.utilization(device)
            except Exception:  # NVML unavailable (container without driver libs, etc.)
                return
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            if self._active.is_set():
                self.samples.append(torch.cuda.utilization(self.device))

    def start(self):
        """Begin recording (call after warmup/compile so they don't skew the mean)."""
        self._active.set()

    def pause(self):
        self._active.clear()

    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
