import spacy
import string
from typing import List

# Load the spacy model once to improve performance
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

def read_file(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def extract_sentences(text: str) -> List[str]:
    """Extracts sentences from a given text using spaCy."""
    doc = NLP(text.lower())
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def remove_punctuation(sentence: str) -> str:
    """Removes punctuation and newline characters from a sentence."""
    return sentence.translate(str.maketrans("", "", string.punctuation)).replace("\n", " ")

def process_text_file(file_path: str) -> List[str]:
    """
    Reads a file, extracts sentences, and cleans them by removing punctuation.

    Args:
        file_path (str): The path to the text file.

    Returns:
        List[str]: A list of cleaned sentences.
    """
    content = read_file(file_path)
    sentences = extract_sentences(content)
    cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences]
    return cleaned_sentences