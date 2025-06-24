import wave
import tempfile
from musearoo.core.precision_timing_handler import PrecisionTimingHandler


def _create_dummy_wav(path: str):
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b'\x00\x00' * 44100)  # 1 second of silence


def test_analyze_wav():
    handler = PrecisionTimingHandler()
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        _create_dummy_wav(tmp.name)
        meta = handler.analyze_input_timing(tmp.name)
        assert meta.total_duration_seconds == 1.0
        assert meta.sample_rate == 44100
