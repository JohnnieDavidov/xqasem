# XQASem

XQASem is a small Python package for extracting predicate-centered question-answer structures from text. It currently includes presets for French, Russian, and Hebrew models hosted on Hugging Face.

The repository contains code only. Model weights should stay on Hugging Face rather than in Git.

This package accompanies the paper [Effective QA-Driven Annotation of Predicate-Argument Relations Across Languages](https://aclanthology.org/2026.eacl-long.112/).

## Installation

After the package is published to PyPI:

```bash
pip install xqasem
```

Install the spaCy pipelines you plan to use:

```bash
python -m spacy download fr_core_news_md
python -m spacy download ru_core_news_sm
```

Hebrew uses `spacy-stanza`; on first use, Stanza may need to download its Hebrew resources.

### Installation from source

```bash
git clone https://github.com/JohnnieDavidov/xqasem.git
cd xqasem
pip install -e .
```

## Basic Usage

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

After installing the package:

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

## Smoke Test

To check that the installed package, spaCy pipelines, model loading, and output writing work end to end, run:

```bash
python scripts/smoke_test_languages.py --langs fr ru he --output-dir outputs/smoke
```

For a faster sanity check on one language:

```bash
python scripts/smoke_test_languages.py --langs fr --max-items 2
```

The smoke test writes one CSV per language and prints the number of rows produced. It is intended for local validation and is not run as part of the lightweight unit tests.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Large model files, checkpoints, generated outputs, and local experiment directories are intentionally ignored by `.gitignore`.

## Publishing to PyPI

The public package name is `xqasem`. After updating the version in
`pyproject.toml` and `xqasem/__init__.py`, build and publish with:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine upload dist/*
```

After the first PyPI release, users can install the package with:

```bash
pip install xqasem
```

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
