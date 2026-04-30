import inspect
from typing import Any, Dict, List, Union

import spacy
import pandas as pd
from spacy.tokens import Doc
from .argument_detection import XQasemArgumentParser
from .presets import DEFAULT_MODELS, DEFAULT_SPACY_MODELS
from tqdm import tqdm


def convert_to_dataframe(
    rows_predictions: List[Dict[str, str]],
) -> pd.DataFrame:
    """
    Convert flat prediction dictionaries into the public DataFrame schema.
    """
    
    df = pd.DataFrame(rows_predictions, columns=[
        "sentence",
        "predicate",
        "predicate_type",
        "question",
        "answer",
    ])
    return df

def print_rows_predictions(row_predictions: List[Dict[str, Any]]
    ) -> None:
    """
    Print flattened predictions in a readable format.
    """

    for row in row_predictions:
        print(f"sentence: {row['sentence']}")
        print(f"predicate: {row['predicate']}")
        print(f"predicate_type: {row['predicate_type']}")
        print(f"question: {row['question']}")
        print(f"answer: {row['answer']}")
        print("-" * 60)


def spacy_analyze(sentences: List[List[str]], nlp: spacy.Language, verbose: bool = False):
    """Run a spaCy pipeline over pre-tokenized sentences.

    The input sentences are converted to ``Doc`` objects first so spaCy keeps
    the provided token boundaries while still running the pipeline components.
    """
    # no need to count everything to determine if should disable if this is a huge payload anyway.
    num_tokens = sum(len(tokens) for tokens in sentences[:500])
    # the default is not to disable, however if payload is small disable anyway.
    disable_print = (not verbose) or (num_tokens < 500)
    docs = [Doc(words=tokens, vocab=nlp.vocab) for tokens in sentences]
    docs = list(tqdm(nlp.pipe(docs), disable=disable_print, desc="Running spacy...", total=len(sentences)))
    return docs


class XQasemParser:
    """High-level parser that handles tokenization, predicate detection, and QA generation."""

    @classmethod
    def from_pretrained(cls,
                        arg_parser_path: str,
                        spacy_lang: str,
                        is_adapter: bool = True,
                        verbose: bool = False,
                        **kwargs
    ):
        """Load spaCy/Stanza analysis and the QASem argument parser.

        Parameters
        ----------
        arg_parser_path
            Local model path or Hugging Face model identifier.
        spacy_lang
            spaCy pipeline name, or ``"he"`` to use ``spacy-stanza`` Hebrew.
        is_adapter
            If True, load the QASem model as a PEFT adapter.
        verbose
            If True, show progress bars and generation progress logs.
        **kwargs
            Additional arguments forwarded to ``XQasemArgumentParser``.
        """
        if spacy_lang == "he":
            import spacy_stanza
            import torch

            if not getattr(torch.load, "_xqasem_patched", False):
                _old_load = torch.load

                def patched_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return _old_load(*args, **kwargs)

                patched_load._xqasem_patched = True
                torch.load = patched_load

            nlp = spacy_stanza.load_pipeline(spacy_lang)
        else:
            nlp = spacy.load(spacy_lang)

        parser_cls = XQasemArgumentParser
        parser_params = list(inspect.signature(parser_cls).parameters)
        parser_kwargs = {
            k: kwargs.get(k) for k in dict(kwargs)
            if k in parser_params
        }
        if "verbose" in parser_params:
            parser_kwargs["verbose"] = verbose

        arg_parser = parser_cls.from_pretrained(arg_parser_path, language=nlp.lang, is_adapter=is_adapter, **parser_kwargs)
        return cls(arg_parser, nlp, verbose=verbose)

    @classmethod
    def from_language(cls, language: str, **kwargs):
        """Load one of the released language presets.

        Parameters
        ----------
        language
            One of ``"fr"``, ``"ru"``, or ``"he"``.
        **kwargs
            Additional arguments forwarded to :meth:`from_pretrained`.
        """
        language = language.lower()
        if language not in DEFAULT_MODELS:
            supported = ", ".join(sorted(DEFAULT_MODELS))
            raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported}.")

        kwargs.setdefault("is_adapter", True)
        return cls.from_pretrained(
            DEFAULT_MODELS[language],
            spacy_lang=DEFAULT_SPACY_MODELS[language],
            **kwargs,
        )

    def __init__(self, arg_parser: XQasemArgumentParser, spacy_lang: spacy.Language, verbose: bool = False):
        """Create a parser from an argument parser and a spaCy language pipeline."""
        self.arg_parser = arg_parser
        self._nlp = spacy_lang
        self.verbose = verbose

    def _normalize_input(self, sentences, is_pretokenized: bool) -> List[Doc]:
        """Normalize user input into analyzed spaCy Docs.

        Accepts a single string, a list of strings, a tokenized sentence, or a
        list of tokenized sentences.
        """
        if not sentences:
            raise ValueError("sentences must be a non-empty string or list")
        out_sentences = sentences
        is_str = isinstance(sentences, str)
        is_list_of_str = isinstance(sentences, list) and isinstance(sentences[0], str)
        is_list_of_list = isinstance(sentences, list) and isinstance(sentences[0], list)
        if is_list_of_list:
            # this must be pre-tokenized
            is_pretokenized = True
        elif is_str:
            # this must be a single sentence, untokenized
            is_pretokenized = False
            out_sentences = [out_sentences]
        elif is_list_of_str and is_pretokenized:
            # this is a single pre_tokenized sentence
            out_sentences = [out_sentences]
        elif is_list_of_str and not is_pretokenized:
            # this is a list of untokenized sentences
            pass
        # these conditions must hold now:
        if is_pretokenized:
            if not isinstance(out_sentences[0], list):
                raise TypeError("pre-tokenized input must be a list of tokens or a list of token lists")
        else:
            if not isinstance(out_sentences[0], str):
                raise TypeError("sentences must be a string, list of strings, or pre-tokenized input")

        if is_pretokenized:
            docs = spacy_analyze(out_sentences, self._nlp, verbose=self.verbose)
        else:
            # spacy will tokenize and analyze the sentences
            docs = list(tqdm(self._nlp.pipe(out_sentences),
                             desc="Running spacy for initial tokenization",
                             total=len(out_sentences),
                             disable=(not self.verbose) or len(out_sentences) < 100))
        return docs
    
    def __call__(self,
                 sentences: Union[List[str],
                                  str,
                                  List[List[str]],
                                  List[str]],
                 is_pretokenized=False,
                 max_items=None) -> pd.DataFrame:
        """Parse input sentences and return flattened Qasem predictions.

        Parameters
        ----------
        sentences : Union[List[str], str, List[List[str]], List[str]]
            Input text to parse. This may be:
            - a single untokenized sentence string,
            - a list of untokenized sentence strings,
            - a single pre-tokenized sentence,
            - or a list of pre-tokenized sentences.
        is_pretokenized : bool, optional
            If True, the input is assumed to be pre-tokenized. Otherwise, spaCy
            will tokenize and analyze untokenized sentence strings.
        max_items : int, optional
            Maximum number of items to process. If None, process all items.

        Returns
        -------
        pandas.DataFrame
            Flattened prediction rows containing columns such as
            ``sentence``, ``predicate``, ``predicate_type``, ``question``, ``answer``,
        """

        if not sentences:
            return convert_to_dataframe([])
        
        docs = self._normalize_input(sentences, is_pretokenized)
        
        rows_predictions = self.arg_parser(docs, max_items=max_items)
        # print_rows_predictions(rows_predictions)
        
        df = convert_to_dataframe(rows_predictions)
        return df
