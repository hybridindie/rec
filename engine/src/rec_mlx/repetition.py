import numpy as np


class RepetitionGuard:
    """Detects a generation stuck repeating the same tokens exactly.

    For each period p up to max_period, track how many consecutive tokens equal the
    token p positions earlier. A run of max(min_run, 2p) means the last stretch is one
    block repeated at least min_run tokens and at least twice. Calibrated on 1,363
    agent-loop generations: it caught every runaway (periods 8-576) and nothing else.
    """

    def __init__(self, *, max_period=1024, min_run=512):
        self.max_period = max_period
        self.period = None
        self._history = np.full(max_period, -1, np.int64)  # _history[i]: token i+1 back
        self._runs = np.zeros(max_period, np.int64)  # _runs[i]: run for period i+1
        self._needed = np.maximum(min_run, 2 * np.arange(1, max_period + 1))

    def add(self, token):
        """Record the next token; True once the output is looping."""
        self._runs = np.where(self._history == token, self._runs + 1, 0)
        self._history = np.roll(self._history, 1)
        self._history[0] = token
        looping = np.flatnonzero(self._runs >= self._needed)
        if looping.size:
            self.period = int(looping[0]) + 1
            return True
        return False
