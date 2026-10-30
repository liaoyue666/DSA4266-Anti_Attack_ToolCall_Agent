"""AgentDojo tool-call auditing pilot."""

def __getattr__(name: str):
    # Offline modeling must not require the optional AgentDojo execution stack.
    if name == "FixedDecisionGate":
        from .gate import FixedDecisionGate

        return FixedDecisionGate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["FixedDecisionGate"]
