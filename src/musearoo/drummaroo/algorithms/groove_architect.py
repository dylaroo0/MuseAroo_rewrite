"""Basic groove architect used for tests."""
from typing import List


def basic_groove(bpm: float, bars: int = 1) -> List[float]:
    """Return kick drum beat times for a simple four-on-the-floor groove."""
    beat_dur = 60.0 / bpm
    pattern = []
    for bar in range(bars):
        start = bar * 4 * beat_dur
        for beat in range(4):
            pattern.append(start + beat * beat_dur)
    return pattern
