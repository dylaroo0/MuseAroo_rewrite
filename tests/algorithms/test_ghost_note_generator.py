from musearoo.drummaroo.algorithms.ghost_note_generator import add_ghost_notes


def test_adds_ghost_notes():
    pattern = [(0.0, 100), (1.0, 100)]
    augmented = add_ghost_notes(pattern, intensity=1.0)
    # with intensity=1.0 a ghost note is inserted after each beat
    assert len(augmented) == 4
    assert augmented[1][0] == 0.5
