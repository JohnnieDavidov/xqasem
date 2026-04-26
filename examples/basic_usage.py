from xqasem import XQasemParser


def main() -> None:
    parser = XQasemParser.from_pretrained(
        "YonatanDavidov/qasem-fr-claire-lora",
        spacy_lang="fr_core_news_md",
        is_adapter=True,
    )

    sentences = [
        "Les développeurs ont expliqué pourquoi la mise à jour avait provoqué des pannes inattendues du service."
    ]
    dataframe = parser(sentences)
    print(dataframe.to_string(index=False))


if __name__ == "__main__":
    main()
