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

