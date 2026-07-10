# Copyright (c) Microsoft. All rights reserved.


def test_top_level_exports_do_not_include_deprecated_symbols() -> None:
    """Ensure deprecated compatibility exports are not re-exported by package root."""

    import agentlightning

    assert not hasattr(agentlightning, "AgentLightningClient")
    assert not hasattr(agentlightning, "AgentLightningServer")
    assert not hasattr(agentlightning, "configure_logger")
    assert not hasattr(agentlightning, "GenericResponse")
    assert not hasattr(agentlightning, "SpanNames")
    assert not hasattr(agentlightning, "SpanAttributeNames")
