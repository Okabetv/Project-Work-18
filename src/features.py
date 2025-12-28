import re

def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zàèéìòù0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text