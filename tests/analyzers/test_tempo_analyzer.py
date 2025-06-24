from musearoo.analyzers.tempo_analyzer import TempoAnalyzer


def test_tempo_analyzer_returns_mock_data():
    analyzer = TempoAnalyzer()
    result = analyzer.analyze("dummy.wav")
    assert result["bpm"] == 120.0
