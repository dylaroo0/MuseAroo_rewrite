"""Simple polyrhythm generator."""
from typing import List, Tuple


def generate_polyrhythm(main_meter: Tuple[int, int], overlay_meter: Tuple[int, int], bars: int = 1) -> List[float]:
    """Return beat positions for an overlay meter aligned with the main meter."""
    main_beats, _ = main_meter
    overlay_beats, _ = overlay_meter

    beat_duration = 1.0  # generic beat unit
    bar_duration = main_beats * beat_duration
    overlay_interval = bar_duration / overlay_beats

    pattern = []
    for bar in range(bars):
        start = bar * bar_duration
        for i in range(overlay_beats):
            pattern.append(start + i * overlay_interval)
    return pattern
