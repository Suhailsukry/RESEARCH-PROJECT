from .bilstm_ctc import BiLSTMCTC
from .fallback import WindowClassifier

_MODELS = {"bilstm_ctc": BiLSTMCTC, "fallback_window": WindowClassifier}


def build_model(model_type: str, kwargs: dict):
    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type {model_type!r}; options: {sorted(_MODELS)}")
    return _MODELS[model_type](**kwargs)
