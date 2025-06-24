from musearoo.drummaroo.algorithms.groove_architect import basic_groove


def test_basic_groove():
    pattern = basic_groove(120, bars=1)
    assert len(pattern) == 4
    assert abs(pattern[1] - 0.5) < 1e-6
