from musearoo.drummaroo.algorithms.polyrhythm_engine import generate_polyrhythm


def test_generate_polyrhythm():
    pattern = generate_polyrhythm((4, 4), (3, 4), bars=1)
    assert len(pattern) == 3
    assert abs(pattern[1] - 4/3) < 1e-6
