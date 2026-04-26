"""Built-in language presets for released XQASem models."""

DEFAULT_MODELS = {
    "fr": "YonatanDavidov/qasem-fr-claire-lora",
    "ru": "YonatanDavidov/qasem-ru-sambalingo-lora",
    "he": "YonatanDavidov/qasem-he-dictalm2-lora",
}

DEFAULT_SPACY_MODELS = {
    "fr": "fr_core_news_md",
    "ru": "ru_core_news_sm",
    "he": "he",
}

DEFAULT_SENTENCES = {
    "fr": [
        "Les développeurs ont expliqué pourquoi la mise à jour avait provoqué des pannes inattendues du service."
    ],
    "ru": [
        "Разработчики объяснили, почему обновление привело к неожиданным сбоям в работе сервиса."
    ],
    "he": [
        "המפתחים הסבירו מדוע העדכון גרם לתקלות בלתי צפויות בשירות."
    ],
}
