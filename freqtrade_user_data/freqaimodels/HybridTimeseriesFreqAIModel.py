"""Expose the custom FreqAI model with a local subclass.

Freqtrade's resolver only considers classes whose __module__ matches this file's
module name. Therefore, subclass the core model so discovery works.
"""
try:
    from hybrid_lstm_transformer_crypto import (
        HybridTimeseriesFreqAIModel_tinhn as CoreHybridTimeseriesFreqAIModel,
    )
except Exception:
    # Backward-compat: fallback to old class if present
    from hybrid_lstm_transformer_crypto import (  # type: ignore
        HybridTimeseriesFreqAIModel as CoreHybridTimeseriesFreqAIModel,  # noqa: F401
    )


class HybridTimeseriesFreqAIModel(CoreHybridTimeseriesFreqAIModel):
    """Backward-compatible alias. Prefer `HybridTimeseriesFreqAIModel_tinhn`."""
    pass
