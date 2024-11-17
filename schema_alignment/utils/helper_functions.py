import re
import json
import unicodedata

from langdetect import detect, DetectorFactory #
from deep_translator import GoogleTranslator #

DetectorFactory.seed = 0


def open_dataset(path) -> tuple[dict, dict]:
    """Load the classes and properties of the database.

    Args:
        path (str): The directory, where the files are.

    Returns:
        tuple: The loaded classes and properties.

    """
    with open(f'{path}/classes.json') as f_classes, \
        open(f'{path}/properties.json') as f_properties:

        classes = json.load(f_classes)
        properties = json.load(f_properties)

    return classes, properties


def camel_case(s):
    words = s.split()  # Split by spaces
    # Capitalize all words except the first and join them back
    return words[0] + ''.join(word.capitalize() for word in words[1:])


def normalize_text(text):
    """Normalize unicode characters to canonical form."""
    return unicodedata.normalize('NFKC', text)


def is_english(text):
    """Detect if the given text is in English."""
    try:
        return detect(text) == 'en'
    except:
        return False


def has_mixed_scripts(text):
    """Detect if the text contains mixed scripts (e.g., Latin and Cyrillic)."""
    latin = re.compile(r'[a-zA-Z]')
    cyrillic = re.compile(r'[\u0400-\u04FF]')

    contains_latin = bool(latin.search(text))
    contains_cyrillic = bool(cyrillic.search(text))

    return contains_latin and contains_cyrillic


def translate_to_english(text):
    """Translate non-English text to English using deep_translator."""
    text = normalize_text(text)  # Normalize the text first

    # Skip if the text has mixed scripts (e.g., both Latin and Cyrillic)
    if has_mixed_scripts(text):
        return None

    if is_english(text):
        return text  # If already in English, return the original text
    else:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return None  # Return None if translation fails