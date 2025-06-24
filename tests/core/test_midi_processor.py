import tempfile
from musearoo.core.midi_processor import load_midi
from mido import Message, MidiFile, MidiTrack


def _create_dummy_midi(path: str):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('note_on', note=60, velocity=64, time=0))
    track.append(Message('note_off', note=60, velocity=64, time=480))
    mid.save(path)


def test_load_midi():
    with tempfile.NamedTemporaryFile(suffix='.mid') as tmp:
        _create_dummy_midi(tmp.name)
        events = load_midi(tmp.name)
        assert len(events) == 1
        assert events[0].note == 60
