import logging
import sys
from logging.handlers import RotatingFileHandler
from constants import *
import Levenshtein
from dotenv import load_dotenv

load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_names(response):
    names = response.split("$")
    return list(filter(lambda x: len(x) > 1, names))


def person_prompt(entity):
    name = entity[CONTEXTUALISING_ATTRIBUTES][Attribute.NAME.value]  # get the value of the first contextrualising attribute (movie title)
    prompt = f"What is the full name of the {entity['entity']} with the following associated information?:"
    for concept, value in tuple(entity[CONTEXTUALISING_ATTRIBUTES].items())[1:]:
        prompt += f"\n- {concept.replace('_', ' ')}: {value}"
    for concept_class in entity[TARGET_ATTRIBUTES]:
        for concept, value in entity[TARGET_ATTRIBUTES][concept_class].items():
            prompt += f"\n- {concept.replace('_', ' ')}: {value}"
    prompt += "\n"
    prompt += "You must respond with just the name of the person. Do not produce any other text."
    return prompt

def name_match(candidate, responses):
    for i in responses:
        if name_similarity(candidate, i):
            return True
    return False


def normalize_string(s):
    """Convert string to lowercase and remove non-alphanumeric characters."""
    return ''.join(c for c in s if c.isalnum()).lower()


def tokenize_string(s):
    """Split string into tokens."""
    return s.split()


def is_abbreviation(abbr, word):
    """Check if `abbr` is an abbreviation of `word`."""
    return word.startswith(abbr)


def name_similarity(name1, name2):
    """Calculate similarity score between two names."""
    # Normalizing the names
    norm_name1 = normalize_string(name1)
    norm_name2 = normalize_string(name2)

    # Tokenizing the names
    tokens1 = tokenize_string(norm_name1)
    tokens2 = tokenize_string(norm_name2)

    # Initial match based on abbreviations
    for token1 in tokens1:
        for token2 in tokens2:
            if is_abbreviation(token1, token2) or is_abbreviation(token2, token1):
                return True

    # Using Levenshtein distance as a similarity metric
    distance = Levenshtein.distance(norm_name1, norm_name2)
    max_len = max(len(norm_name1), len(norm_name2))
    similarity = (max_len - distance) / max_len

    return similarity > 0.8  # Threshold can be adjusted


