import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)