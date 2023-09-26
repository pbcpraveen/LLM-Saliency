import itertools
import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import json
import openai
import pickle
import dotenv

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import propmts
from common_utils import *
from constants import *
import re


def remove_numeric_bullets(sentence):
    # The regex pattern is looking for a number followed by a dot and space at the start of a string
    return re.sub(r"^\s*\d+\.\s*", '', sentence)


def get_prompt_generator_prompt(entity, target_attribute):
    prompt = (f"I want to prompt a Large Language Model to recall the {target_attribute} of a {entity[ENTITY]} by "
              f"providing the following attributes of the person ")
    for concept in entity[CONTEXTUALISING_ATTRIBUTES]:
        prompt += f"\n- {concept}"
    prompt += "\n"
    prompt += (f"Provide 10 best prompts with varying lengths. In your response you must have 3 short prompts, "
               f"3 medium length prompts and 4 long prompts. Further, Your prompts must have all "
               f"{len(entity[CONTEXTUALISING_ATTRIBUTES])} attributes of the {entity[ENTITY]}. You must "
               f"just list the prompts separated by a $ in your response and no other text.")
    return prompt


def get_entity_prompt(meta: dict, entity: dict, template: str, target_attribute: str):
    contextualising_attributes = entity[CONTEXTUALISING_ATTRIBUTES]
    print(contextualising_attributes)
    for i in meta[CONTEXTUALISING_ATTRIBUTES]:
        if contextualising_attributes[i] is not None:
            template = template.replace(i, contextualising_attributes[i].replace('\"', ''))
    template += '\n'
    template += f'You must only output the {target_attribute.replace("_", " ")} in your response.'
    return template.replace("[", "").replace("]", "")


def get_score(df, index):
    filtered = df[df[PROMPT_INDEX_COLUMN] == index]
    ground_truth = filtered[GROUND_TRUTH].to_list()
    response = filtered[GPT_4_RESPONSE].to_list()
    result = [str(response[i]) == str(ground_truth[i]) for i in range(len(response))]
    accuracy = sum(result) / len(result)
    return accuracy
