[![PyPI version](https://img.shields.io/pypi/v/xqasem.svg)](https://pypi.org/project/xqasem/)

# XQASem

XQASem is a Python package for extracting predicate-centered question-answer structures from text. It currently includes presets for French, Russian, and Hebrew models hosted on HuggingFace.

This package accompanies the paper [Effective QA-Driven Annotation of Predicate-Argument Relations Across Languages](https://aclanthology.org/2026.eacl-long.112/).

## Installation

```bash
pip install xqasem
```

Install the spaCy pipelines you plan to use:

```bash
python -m spacy download fr_core_news_md
python -m spacy download ru_core_news_sm
```

Hebrew uses `spacy-stanza`; on first use, Stanza may need to download its Hebrew resources.

## Environment

This project was tested with the following setup:

- Python 3.10.20
- torch 2.6.0 (CUDA 12.4)
- transformers 4.57.1
- spaCy 3.7.5

## Requirements

- Python 3.10+
- transformers >= 4.50
- spaCy >= 3.7
- torch >= 2.0

Note: GPU is recommended for efficient inference.

### Installation from source

```bash
git clone https://github.com/JohnnieDavidov/xqasem.git
cd xqasem
pip install -e .
```

## Basic Usage

### Example

```python
from xqasem import XQasemParser

parser = XQasemParser.from_language("fr")

sentences = [
    "Les experts ont souligné que le nouvel algorithme accélère considérablement le traitement des requêtes complexes."
]

df = parser(sentences)

print(df)
```

Example output:

| sentence | predicate | predicate_type | question | answer |
| --- | --- | --- | --- | --- |
| Les experts ont souligné que le nouvel algorithme accélère considérablement le traitement des requêtes complexes. | souligné | verb | qui a souligné quelque chose? | Les experts |
| Les experts ont souligné que le nouvel algorithme accélère considérablement le traitement des requêtes complexes. | accélère | verb | qu'est-ce qui accélère quelque chose? | le nouvel algorithme |
| Les experts ont souligné que le nouvel algorithme accélère considérablement le traitement des requêtes complexes. | accélère | verb | qu'est-ce que quelque chose accélère? | le traitement des requêtes complexes |

The built-in language presets are:

```python
parser_fr = XQasemParser.from_language("fr")
parser_ru = XQasemParser.from_language("ru")
parser_he = XQasemParser.from_language("he")
```

You can also load an explicit Hugging Face model:

```python
parser = XQasemParser.from_pretrained(
    "YonatanDavidov/qasem-fr-claire-lora",
    spacy_lang="fr_core_news_md",
    is_adapter=True,
)
```

## Command Line

After installation, you can run:

```bash
xqasem-parse --lang fr --output outputs/fr.csv "Les experts ont souligné que le nouvel algorithme accélère le traitement."
```

You can also run the script directly:

```bash
python scripts/run_parser.py --lang fr --output outputs/fr.csv
```

## Model Presets

| Language | Model | spaCy pipeline |
| --- | --- | --- |
| French | [`YonatanDavidov/qasem-fr-claire-lora`](https://huggingface.co/YonatanDavidov/qasem-fr-claire-lora) | `fr_core_news_md` |
| Russian | [`YonatanDavidov/qasem-ru-sambalingo-lora`](https://huggingface.co/YonatanDavidov/qasem-ru-sambalingo-lora) | `ru_core_news_sm` |
| Hebrew | [`YonatanDavidov/qasem-he-dictalm2-lora`](https://huggingface.co/YonatanDavidov/qasem-he-dictalm2-lora) | `he` via `spacy-stanza` |
| Hebrew full model | [`YonatanDavidov/qasem-he-dictalm2-full`](https://huggingface.co/YonatanDavidov/qasem-he-dictalm2-full) | `he` via `spacy-stanza` |


## Citation

If you use this code or the released models, please cite:

```bibtex
@inproceedings{davidov-etal-2026-effective,
    title = "Effective {QA}-Driven Annotation of Predicate{--}Argument Relations Across Languages",
    author = "Davidov, Jonathan  and
      Slobodkin, Aviv  and
      Klein, Shmuel Tomi  and
      Tsarfaty, Reut  and
      Dagan, Ido  and
      Klein, Ayal",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.112/",
    doi = "10.18653/v1/2026.eacl-long.112",
    pages = "2484--2502",
    ISBN = "979-8-89176-380-7"
}
```

## Acknowledgments

* [Ayal Klein](https://github.com/kleinay)
* [Paul Roit](https://github.com/plroit)

This implementation builds on ideas and code structure from Paul Roit's
[`qasem_parser`](https://github.com/plroit/qasem_parser), which provides the
original English QA-Sem parsing framework that inspired this multilingual
extension.
