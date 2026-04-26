from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, TYPE_CHECKING, Union

from .argument_correction import validate_and_fix_answer

if TYPE_CHECKING:
    # Imported only for type hints; avoids requiring spaCy at annotation runtime.
    import spacy

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_LENGTH = 96
logger = logging.getLogger(__name__)


class XQasemArgumentParser:
    """Generate QASem question-answer pairs for highlighted verbal predicates."""

    def __init__(
        self,
        model,
        tokenizer,
        language: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_new_tokens: int = _DEFAULT_MAX_LENGTH,
        do_sample: bool = True,
        temperature: float = 0.1,
        top_p: float = 0.9,
        verbose: bool = False,
    ):
        """Create a parser from an already loaded causal language model.

        Parameters
        ----------
        model
            A Transformers-compatible causal language model.
        tokenizer
            The tokenizer paired with ``model``.
        language
            ISO-like language code supported by the prompts: ``"fr"``, ``"ru"``,
            or ``"he"``.
        batch_size
            Number of highlighted predicate examples to generate per batch.
        max_new_tokens
            Maximum number of new tokens to generate for each highlighted item.
        do_sample, temperature, top_p
            Generation sampling parameters. By default generation is greedy.
        verbose
            If True, log progress information while generating.
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.language = language.lower()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        """Run :meth:`predict` so parser instances are directly callable."""
        return self.predict(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        path_or_model_name: str,
        language: str,
        is_adapter: bool = True,
        device: str = None,
        dtype=None,
        low_cpu_mem_usage: bool = True,
        load_in_4bit: bool = True,
        **kwargs
    ):
        """Load a model, tokenizer, and parser from a local path or Hub name.

        Parameters
        ----------
        path_or_model_name
            Local model path or Hugging Face model identifier.
        language
            Parser language code: ``"fr"``, ``"ru"``, or ``"he"``.
        is_adapter
            If True, load the model as a PEFT adapter. If False, load it as a
            full causal language model.
        device
            Target device. Defaults to CUDA when available, otherwise CPU.
        dtype
            Optional torch dtype. If omitted on CUDA, bfloat16 or float16 is
            selected automatically.
        low_cpu_mem_usage
            Passed through to Transformers model loading.
        load_in_4bit
            Enable 4-bit quantization on CUDA devices.
        **kwargs
            Additional arguments forwarded to :class:`XQasemArgumentParser`.
        """
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }

        if dtype is None and device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        # Quantization only when running on CUDA
        if device.startswith("cuda") and load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(path_or_model_name)

        if is_adapter:
            model = AutoPeftModelForCausalLM.from_pretrained(
                path_or_model_name,
                **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path_or_model_name,
                **model_kwargs
            )

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.return_dict = True
        
        # Set padding side to left for decoder-only models (like GPT-style models)
        tokenizer.padding_side = 'left'
        return cls(model=model, tokenizer=tokenizer, language=language, **kwargs)
    
    @staticmethod
    def _is_verbal_predicate(tok) -> bool:
        """Return True when a spaCy token should be treated as a verbal predicate."""
        if tok.pos_ != "VERB":
            return False
        if tok.dep_.lower() == "aux":
            return False
        if tok.dep_.lower() == "amod":
            return False
        if not tok.is_alpha:
            return False
        return True

    @staticmethod
    def _normalize_docs_input(docs: Union[spacy.tokens.Doc, List[spacy.tokens.Doc]]) -> List[spacy.tokens.Doc]:
        """Normalize a single spaCy Doc or a list of Docs into a list."""
        if isinstance(docs, list):
            return docs
        return [docs]

    def _get_system_role_by_language(self, language):
        """
        Return the prompt instruction used for the requested language.
        """
        language = language.lower()
        if language == 'he':
            return "עבור משפט ופרדיקט שמודגש בתוכו, תייצר את כל השאלות והתשובות כאשר התשובות שלהן נמצאות בתוך המשפט. התשובות חייבות להיות קטע רציף מתוך המשפט, והשאלה חייבת להכיל את הפרדיקט."
        elif language == 'ru':
            return "Для данного предложения и выделенного в нем предиката, сгенерируйте все вопросы и ответы для данного предиката, при этом ответы должны находиться внутри предложения. Ответы должны быть непрерывным фрагментом предложения, а вопрос должен содержать предикат"
        elif language == 'fr':
            return "Pour une phrase donnée et un prédicat mis en évidence, générez toutes les questions et réponses pour le prédicat donné, tandis que les réponses doivent se trouver à l'intérieur de la phrase. Les réponses doivent être un fragment continu de la phrase, et la question doit contenir le prédicat."
        else:
            raise ValueError("Unsupported language. Use 'he', 'ru', or 'fr'.")

    @classmethod
    def extract_verb_highlighted_sentences(
        cls,
        docs: Union[spacy.tokens.Doc, List[spacy.tokens.Doc]]
    ) -> List[Dict[str, Any]]:
        """
        Extract all VERB predicates from the input docs.
        Each predicate creates a separate example with the predicate highlighted.
        """
        docs = cls._normalize_docs_input(docs)
        results = []

        for doc_idx, doc in enumerate(docs):
            for sent_idx, sent in enumerate(doc.sents):
                verbs = [tok for tok in sent if cls._is_verbal_predicate(tok)]

                for verb in verbs:
                    parts = []
                    
                    for tok in sent:
                        token_text = tok.text
                        
                        if tok.i == verb.i:
                            token_text = f"**{token_text}**"
                        parts.append(token_text + tok.whitespace_)

                    highlighted_sentence = "".join(parts).strip()

                    results.append({
                        "doc_index": doc_idx,
                        "sent_index": sent_idx,
                        "sentence_text": sent.text,
                        "highlighted_sentence": highlighted_sentence,
                        "predicate_text": verb.text,
                        "predicate_lemma": verb.lemma_,
                        "predicate_index_in_doc": verb.i,
                        "predicate_index_in_sentence": verb.i - sent.start,
                        "sentence_tokens": [tok.text for tok in sent],
                    })

        return results
        
    def build_messages(self, highlighted_sentence: str) -> List[Dict[str, str]]:
        """Build chat-template messages for one predicate-highlighted sentence.

        Hebrew combines the instruction and highlighted sentence into one user
        message because the current Hebrew model follows that format more
        reliably. French and Russian use separate system and user messages.
        """
        system_role = self._get_system_role_by_language(self.language)
        
        if self.language == "he":
            # in Hebrew, we combine the system role and user content into one
            # message to avoid issues with the model not properly attending to the system role when it's separate
            combined_content = f"{system_role}\n{highlighted_sentence}"
            return [{"role": "user", "content": combined_content}]
        else:
            # for French and Russian, we can keep the system role and user content 
            # separate as the models seem to handle it fine
            return [
                {"role": "system", "content": system_role},
                {"role": "user", "content": highlighted_sentence}
            ]

    def _get_terminators(self) -> List[int]:
        """Return token ids that should stop generation."""
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [self.tokenizer.eos_token_id]

        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)

        return terminators

    def generate_batch(self, highlighted_sentences: List[str]) -> List[str]:
        """Generate raw model outputs for a batch of highlighted sentences."""
        import torch

        messages = [self.build_messages(sentence) for sentence in highlighted_sentences]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_dict=True,
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        terminators = self._get_terminators()

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "eos_token_id": terminators,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_beams": 1,
            "use_cache": True,
            "return_dict_in_generate": False,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)

        batch_responses = []
        prompt_len = input_ids.shape[-1]

        for output in outputs:
            response = output[prompt_len:]
            generated_text = self.tokenizer.decode(
                response,
                skip_special_tokens=True
            ).strip()
            batch_responses.append(generated_text)

        return batch_responses

    def predict_from_highlighted_items(
        self,
        highlighted_items: List[Dict[str, Any]],
        max_items: int = None
    ) -> List[Dict[str, Any]]:
        """Run generation over extracted predicate items.

        Parameters
        ----------
        highlighted_items
            Items produced by :meth:`extract_verb_highlighted_sentences`.
        max_items
            Optional limit for quick experiments or tests.

        Returns
        -------
        list of dict
            The original item metadata plus a ``model_output`` field.
        """
        predictions = []

        if max_items is not None:
            highlighted_items = highlighted_items[:max_items]

        total_start_time = time.time()
        total_items = len(highlighted_items)

        if total_items == 0:
            return predictions

        if self.verbose:
            logger.info("Processing %s items in batches of %s", total_items, self.batch_size)

        for batch_start in range(0, len(highlighted_items), self.batch_size):
            batch_items = highlighted_items[batch_start : batch_start + self.batch_size]
            batch_sentences = [item["highlighted_sentence"] for item in batch_items]
            
            batch_start_time = time.time()
            batch_outputs = self.generate_batch(batch_sentences)
            batch_time = time.time() - batch_start_time

            items_per_sec = len(batch_items) / max(batch_time, 1e-9)
            progress = (batch_start + len(batch_items)) / total_items * 100

            if self.verbose:
                logger.info(
                    "Batch %s: %s items in %.2fs (%.2f items/sec) - %.1f%% complete",
                    batch_start // self.batch_size + 1,
                    len(batch_items),
                    batch_time,
                    items_per_sec,
                    progress,
                )

            for item, model_output in zip(batch_items, batch_outputs):
                predictions.append({
                    "doc_index": item["doc_index"],
                    "sent_index": item["sent_index"],
                    "sentence_text": item["sentence_text"],
                    "highlighted_sentence": item["highlighted_sentence"],
                    "predicate_text": item["predicate_text"],
                    "predicate_lemma": item["predicate_lemma"],
                    "predicate_index_in_doc": item["predicate_index_in_doc"],
                    "predicate_index_in_sentence": item["predicate_index_in_sentence"],
                    "sentence_tokens": item["sentence_tokens"],
                    "model_output": model_output,
                })

        total_time = time.time() - total_start_time
        overall_items_per_sec = total_items / max(total_time, 1e-9)
        if self.verbose:
            logger.info("Total processing time: %.2fs (%.2f items/sec)", total_time, overall_items_per_sec)

        return predictions
    
    
    @staticmethod
    def _split_by_arrow(text: str) -> tuple:
        """Split text by arrow (→ or ->) and return (question, answer) or (None, None)."""
        if "→" in text:
            parts = text.split("→", 1)
        elif "->" in text:
            parts = text.split("->", 1)
        else:
            return None, None
        
        if len(parts) != 2:
            return None, None
        return parts[0].strip(), parts[1].strip()
    
    def _clean_qa_pair(self, question: str, answer: str) -> tuple:
        """Clean minor formatting artifacts from one parsed QA pair."""
        # Remove leading dot from Hebrew questions
        if self.language == "he" and question.startswith("·"):
            question = question[1:].strip()
        
        # Remove surrounding quotes from answer
        if len(answer) >= 2 and (
            (answer[0] == "'" and answer[-1] == "'") or
            (answer[0] == '"' and answer[-1] == '"')
        ):
            answer = answer[1:-1].strip()
        
        return question, answer
    
    def _parse_model_output_to_qa_pairs(
        self, 
        model_output: str
    ) -> List[Dict[str, str]]:
        """
        Parse model output of the form:
        question -> 'answer'
        or
        question → 'answer'

        For Hebrew: supports both line-separated and pipe-separated formats
        """
        if not model_output or not model_output.strip():
            return []

        qa_pairs = []
        
        # For Hebrew: handle pipe-separated format
        if self.language == "he" and "|" in model_output:
            parts = [part.strip() for part in model_output.split("|") if part.strip()]
            items_to_parse = parts
        else:
            # Line-based parsing for other languages
            items_to_parse = [line.strip() for line in model_output.splitlines() if line.strip()]
        
        # Parse each item
        for item in items_to_parse:
            question, answer = self._split_by_arrow(item)
            if question is None or answer is None:
                continue
            
            question, answer = self._clean_qa_pair(question, answer)
            
            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })

        return qa_pairs

    def flatten_predictions_to_rows(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert model predictions into flat rows for CSV.
        Currently predicate_type is always 'verb'.
        """
        rows_predictions = []

        for item in predictions:
            sentence = item.get("sentence_text", "")
            predicate = item.get("predicate_text", "")
            model_output = item.get("model_output", "")

            qa_pairs = self._parse_model_output_to_qa_pairs(model_output)
            
            for qa in qa_pairs:
                answer_validation = validate_and_fix_answer(sentence, qa["answer"])

                rows_predictions.append({
                    "sentence": sentence,
                    "predicate": predicate,
                    "predicate_type": "verb",
                    "question": qa["question"],
                    "answer": answer_validation["fixed_answer"],
                })

        return rows_predictions

    def predict(
        self,
        docs: Union[spacy.tokens.Doc, List[spacy.tokens.Doc]],
        max_items: int = None
    ) -> List[Dict[str, Any]]:
        """Extract predicates, generate QA pairs, and return flat output rows."""
        highlighted_items = self.extract_verb_highlighted_sentences(docs)
        predictions = self.predict_from_highlighted_items(highlighted_items, max_items=max_items)
        rows_predictions = self.flatten_predictions_to_rows(predictions=predictions)
        return rows_predictions
