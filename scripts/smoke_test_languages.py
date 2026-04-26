import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xqasem import XQasemParser
from xqasem.cli import DEFAULT_MODELS, DEFAULT_SENTENCES, DEFAULT_SPACY_MODELS


def parse_args() -> argparse.Namespace:
    """Parse arguments for the multilingual smoke test."""
    parser = argparse.ArgumentParser(description="Run XQASem smoke tests for one or more languages.")
    parser.add_argument("--langs", nargs="+", choices=sorted(DEFAULT_MODELS), default=sorted(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default="outputs/smoke")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--full-model", action="store_true", help="Load presets as full models instead of adapters.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run each selected language preset and save its output as CSV."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        print(f"Running {lang}...")
        parser = XQasemParser.from_pretrained(
            DEFAULT_MODELS[lang],
            spacy_lang=DEFAULT_SPACY_MODELS[lang],
            is_adapter=not args.full_model,
            verbose=args.verbose,
        )
        dataframe = parser(DEFAULT_SENTENCES[lang], max_items=args.max_items)
        output_path = output_dir / f"{lang}.csv"
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{lang}: wrote {len(dataframe)} rows to {output_path}")


if __name__ == "__main__":
    main()
