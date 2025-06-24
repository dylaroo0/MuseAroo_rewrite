from musearoo.core.context_integrator import create_basic_integrator, CompleteRooContext


def test_context_summary_defaults():
    integrator = create_basic_integrator()
    ctx = CompleteRooContext()
    summary = integrator.get_context_summary(ctx)
    assert "total_features" in summary
