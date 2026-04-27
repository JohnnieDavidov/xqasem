from .argument_detection import XQasemArgumentParser
from .presets import DEFAULT_MODELS, DEFAULT_SPACY_MODELS
from .xqasem_parsing import XQasemParser
from importlib.metadata import version

__version__ = version("xqasem")

__all__ = ["DEFAULT_MODELS", "DEFAULT_SPACY_MODELS", "XQasemArgumentParser", "XQasemParser"]
