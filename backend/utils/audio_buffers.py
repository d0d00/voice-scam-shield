from __future__ import annotations

import numpy as np


class SlidingWindowBuffer:
    """A fixed-capacity circular buffer for mono float32 audio samples.

    - Stores up to capacity_samples at 16-bit float32 values in [-1, 1].
    - push() appends samples (wraps on overflow) with O(n) copies on wrap.
    - get_recent(n) returns a contiguous np.ndarray copy of the last n samples.
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self.capacity = int(capacity_samples)
        self._buffer = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos = 0
        self._size = 0

    def clear(self) -> None:
        self._write_pos = 0
        self._size = 0

    def push(self, samples: np.ndarray) -> None:
        if samples is None:
            return
        if not isinstance(samples, np.ndarray):
            samples = np.asarray(samples, dtype=np.float32)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        n = samples.shape[0]
        if n <= 0:
            return

        # Write with wrap handling
        end_pos = self._write_pos + n
        if end_pos <= self.capacity:
            self._buffer[self._write_pos:end_pos] = samples
        else:
            first = self.capacity - self._write_pos
            self._buffer[self._write_pos :] = samples[:first]
            self._buffer[: end_pos % self.capacity] = samples[first:]

        self._write_pos = end_pos % self.capacity
        self._size = min(self.capacity, self._size + n)

    def get_recent(self, n: int) -> np.ndarray:
        if self._size == 0 or n <= 0:
            return np.zeros(0, dtype=np.float32)
        n = int(min(n, self._size))
        start = (self._write_pos - n) % self.capacity
        if start + n <= self.capacity:
            return self._buffer[start : start + n].copy()
        first = self.capacity - start
        out = np.empty(n, dtype=np.float32)
        out[:first] = self._buffer[start:]
        out[first:] = self._buffer[: n - first]
        return out

    def size(self) -> int:
        return self._size


