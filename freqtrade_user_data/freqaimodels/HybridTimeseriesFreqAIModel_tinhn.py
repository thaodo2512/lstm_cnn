"""Expose the custom FreqAI model with a local subclass for discovery.

Freqtrade's resolver considers classes whose __module__ matches this file's
module name. We subclass the core model so discovery works with this name.
"""
from hybrid_lstm_transformer_crypto import (
    HybridTimeseriesFreqAIModel_tinhn as CoreHybridTimeseriesFreqAIModel,
)


class HybridTimeseriesFreqAIModel_tinhn(CoreHybridTimeseriesFreqAIModel):
    pass

