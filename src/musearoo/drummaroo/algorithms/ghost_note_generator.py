"""Simple ghost note generator for DrummaRoo."""
from typing import List, Tuple
import random

# A drum pattern is a list of (time, velocity) tuples.

def add_ghost_notes(pattern: List[Tuple[float, int]], *, intensity: float = 0.2) -> List[Tuple[float, int]]:
    """Return a pattern with ghost notes inserted.

    Ghost notes are created halfway between existing notes with a reduced
    velocity. ``intensity`` controls the probability of insertion.
    """
    if not pattern:
        return []

    pattern = sorted(pattern, key=lambda x: x[0])
    augmented: List[Tuple[float, int]] = []
    for i, (time, vel) in enumerate(pattern):
        augmented.append((time, vel))
        # determine midpoint to next note (or half-beat after last note)
        if i < len(pattern) - 1:
            next_time = pattern[i + 1][0]
            mid = (time + next_time) / 2
        else:
            mid = time + 0.5
        if random.random() < intensity:
            augmented.append((mid, int(vel * 0.4)))
    return sorted(augmented, key=lambda x: x[0])
