import argparse
from pathlib import Path

from . import XQasemParser
from .presets import DEFAULT_MODELS, DEFAULT_SENTENCES, DEFAULT_SPACY_MODELS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the XQASem CLI."""
    parser = argparse.ArgumentParser(description="Run the XQASem parser.")
    parser.add_argument("--lang", choices=sorted(DEFAULT_MODELS), default="fr")
    parser.add_argument("--model", help="Model name or local path. Defaults to the selected language preset.")
    parser.add_argument("--spacy-lang", help="spaCy pipeline name. Defaults to the selected language preset.")
    parser.add_argument("--output", help="Optional CSV output path.")
    parser.add_argument("--full-model", action="store_true", help="Load as a full Transformers model instead of a PEFT adapter.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("sentences", nargs="*", help="Input sentences. If omitted, built-in examples are used.")
    return parser.parse_args()


def main() -> None:
    """Load the selected parser preset, run inference, and print or save output."""
    args = parse_args()
    model_name = args.model or DEFAULT_MODELS[args.lang]
    spacy_lang = args.spacy_lang or DEFAULT_SPACY_MODELS[args.lang]
    sentences = args.sentences or DEFAULT_SENTENCES[args.lang]

    if args.model or args.spacy_lang or args.full_model:
        parser = XQasemParser.from_pretrained(
            model_name,
            spacy_lang=spacy_lang,
            is_adapter=not args.full_model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )
    else:
        parser = XQasemParser.from_language(
            args.lang,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )
    dataframe = parser(sentences)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(dataframe)} rows to {output_path}")
    else:
        print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
