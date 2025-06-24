"""Minimal precision timing handler used for unit tests."""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mido


@dataclass
class TimingMetadata:
    total_duration_seconds: float
    sample_rate: Optional[int] = None
    leading_silence_seconds: float = 0.0
    trailing_silence_seconds: float = 0.0
    file_format: str = ""


class PrecisionTimingHandler:
    """Very small timing analyzer supporting WAV and MIDI files."""

    def analyze_input_timing(self, file_path: str) -> TimingMetadata:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)
        suffix = path.suffix.lower()
        if suffix in {".wav"}:
            return self._analyze_wav(path)
        if suffix in {".mid", ".midi"}:
            return self._analyze_midi(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _analyze_wav(self, path: Path) -> TimingMetadata:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
        return TimingMetadata(duration, sample_rate=rate, file_format="audio")

    def _analyze_midi(self, path: Path) -> TimingMetadata:
        mid = mido.MidiFile(str(path))
        ticks_per_beat = mid.ticks_per_beat or 480
        tempo = 500000  # default microseconds per beat (120 bpm)
        time = 0
        first_note_time = None
        last_note_time = 0
        for msg in mid:
            time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type == "note_on" and msg.velocity > 0:
                if first_note_time is None:
                    first_note_time = time
                last_note_time = time
        duration = time
        leading = first_note_time or 0.0
        trailing = duration - (last_note_time or duration)
        return TimingMetadata(duration, leading_silence_seconds=leading, trailing_silence_seconds=trailing, file_format="midi")

    def ensure_output_timing_match(self, _input: TimingMetadata, output_file: str) -> str:
        # In this simplified version we just return the path as-is
        return output_file
