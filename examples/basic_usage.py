import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xqasem import XQasemParser


def main() -> None:
    parser = XQasemParser.from_language("fr")

    sentences = [
        "Les développeurs ont expliqué pourquoi la mise à jour avait provoqué des pannes inattendues du service."
    ]
    dataframe = parser(sentences)
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
