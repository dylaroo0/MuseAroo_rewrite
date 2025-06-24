"""Utility functions for MIDI processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import mido


@dataclass
class MIDIEvent:
    time: float
    note: int
    velocity: int


def load_midi(file_path: str) -> List[MIDIEvent]:
    mid = mido.MidiFile(file_path)
    events: List[MIDIEvent] = []
    time = 0.0
    tempo = 500000
    for msg in mid:
        time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            events.append(MIDIEvent(time, msg.note, msg.velocity))
    return events
