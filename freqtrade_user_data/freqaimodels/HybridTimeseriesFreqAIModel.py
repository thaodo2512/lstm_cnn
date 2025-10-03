"""Expose the custom FreqAI model with a local subclass.

Freqtrade's resolver only considers classes whose __module__ matches this file's
module name. Therefore, subclass the core model so discovery works.
"""
from hybrid_lstm_transformer_crypto import (
    HybridTimeseriesFreqAIModel as CoreHybridTimeseriesFreqAIModel,
)


class HybridTimeseriesFreqAIModel(CoreHybridTimeseriesFreqAIModel):
    pass
