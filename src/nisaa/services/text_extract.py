"""
Text preprocessing with URL preservation - EXISTING LOGIC INTACT
"""
import re
from typing import Dict, List, Tuple, Optional
from src.nisaa.helpers.logger import logger

class TextPreprocessor:
    """Handles all text preprocessing operations with URL preservation"""

    def __init__(self, config: Optional[Dict[str, bool]] = None):
        self.config = config or {
            "lowercase": True,
            "remove_html": True,
            "preserve_urls": True,
            "remove_extra_whitespace": True,
            "expand_contractions": True,
            "remove_emoji": True,
            "normalize_unicode": True,
        }

        self.chat_words = {
            "FYI": "For Your Information",
            "ASAP": "As Soon As Possible",
            "BRB": "Be Right Back",
            "BTW": "By The Way",
            "OMG": "Oh My God",
            "IMO": "In My Opinion",
            "LOL": "Laugh Out Loud",
            "TTYL": "Talk To You Later",
            "GTG": "Got To Go",
            "IDK": "I Don't Know",
            "TMI": "Too Much Information",
            "IMHO": "In My Humble Opinion",
            "AFAIK": "As Far As I Know",
            "FAQ": "Frequently Asked Questions",
            "TGIF": "Thank God It's Friday",
            "AFK": "Away From Keyboard",
        }

        self.html_pattern = re.compile("<.*?>")
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.whitespace_pattern = re.compile(r"\s+")
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

    def lowercase(self, text: str) -> str:
        return text.lower() if self.config.get("lowercase") else text

    def remove_html_tags(self, text: str) -> str:
        if not self.config.get("remove_html"):
            return text
        return self.html_pattern.sub("", text)

    def extract_and_preserve_urls(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract URLs and replace with semantic placeholders"""
        if not self.config.get("preserve_urls"):
            return self.url_pattern.sub("", text), []

        urls_found = []
        url_matches = list(self.url_pattern.finditer(text))

        processed_text = text
        for i, match in enumerate(reversed(url_matches), 1):
            url = match.group()
            placeholder = f"[URL_REFERENCE_{len(url_matches) - i + 1}]"

            urls_found.insert(
                0,
                {
                    "id": f"URL_{len(url_matches) - i + 1}",
                    "url": url,
                    "placeholder": placeholder,
                    "position": match.start(),
                },
            )

            processed_text = (
                processed_text[: match.start()]
                + placeholder
                + processed_text[match.end() :]
            )

        return processed_text, urls_found

    def remove_extra_whitespace(self, text: str) -> str:
        if not self.config.get("remove_extra_whitespace"):
            return text
        text = self.whitespace_pattern.sub(" ", text)
        return text.strip()

    def expand_chat_words(self, text: str) -> str:
        if not self.config.get("expand_contractions"):
            return text

        words = text.split()
        expanded_words = []

        for word in words:
            upper_word = word.upper()
            if upper_word in self.chat_words:
                expanded_words.append(self.chat_words[upper_word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    def remove_emoji(self, text: str) -> str:
        if not self.config.get("remove_emoji"):
            return text
        return self.emoji_pattern.sub("", text)

    def normalize_unicode(self, text: str) -> str:
        if not self.config.get("normalize_unicode"):
            return text
        import unicodedata
        return unicodedata.normalize("NFKD", text)

    def preprocess(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Apply all preprocessing steps in optimal order"""
        if not text or not isinstance(text, str):
            return "", []

        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.remove_emoji(text)
        text, urls = self.extract_and_preserve_urls(text)
        text = self.expand_chat_words(text)
        text = self.lowercase(text)
        text = self.remove_extra_whitespace(text)

        return text, urls