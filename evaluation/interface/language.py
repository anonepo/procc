from loguru import logger

PREDIFINED_LANGUAGE = {
    "Java": "//",
    "Kotlin": "//",
    "Python": "#",
    "JavaScript": "//",
    "TypeScript": "//",
    "TypeScriptReact": "//",
    "Vue": "//",
    "C": "//",
    "C++": "//",
    "Objective-C++": "//",
    "Swift": "//",
    "SQL": "--",
    "Hive SQL": "--",
}

LANGUAGE_FORMAT = {
    "java": "Java",
    ".java": "Java",
    "kotlin": "Kotlin",
    "kt": "Kotlin",
    ".kt": "Kotlin",
    "python": "Python",
    "py": "Python",
    ".py": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScrispt",
    "ts": "TypeScript",
    "typescriptreact": "TypeScriptReact",
    "vue": "Vue",
    "c": "C",
    "cpp": "C++",
    "c++": "C++",
    "objc": "Objective-C++",
    "objective": "Objective-C++",
    "swift": "Swift",
    "sql": "SQL",
    "hql": "Hive SQL",
}


class LanguageParser:
    __predifined_language = PREDIFINED_LANGUAGE
    __language_format = LANGUAGE_FORMAT

    @classmethod
    def format_language(cls, language: str) -> str:
        language = cls.__language_format.get(language.lower(), language)
        if language not in cls.__predifined_language:
            logger.warning(f"Language {language} is not supported.")
            language = ""
        return language

    @classmethod
    def get_comment(cls, language: str) -> str:
        language = cls.format_language(language)
        comment = cls.__predifined_language.get(language, "//")
        return comment
