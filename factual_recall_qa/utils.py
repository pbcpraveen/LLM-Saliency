import itertools
import sys
import os
from pathlib import Path
import pickle
from random import sample

import numpy as np
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
    # print(contextualising_attributes)
    for i in meta[CONTEXTUALISING_ATTRIBUTES]:
        try:
            if contextualising_attributes[i] is not None:
                    template = template.replace(i, contextualising_attributes[i].replace('\"', ''))
        except:
           template = template.replace(i, "unknown")
    template += '\n'
    template += f'You must only output the {target_attribute.replace("_", " ")} in your response.'
    return template.replace("[", "").replace("]", "")

def score_attribute(concept_class, generation, ground_truth):
    if concept_class == ConceptClass.YEAR.value:
        return str(extract_year(generation)) == str(extract_year(ground_truth))
    else:
        return name_similarity(str(generation), str(ground_truth))

def get_score(df, index, concept_class=ConceptClass.YEAR.value):
    filtered = df[df[PROMPT_INDEX_COLUMN] == index]
    ground_truth = filtered[GROUND_TRUTH].to_list()
    response = filtered[GPT_4_RESPONSE].to_list()
    result = [score_attribute(concept_class,response[i],ground_truth[i]) for i in range(len(response))]
    accuracy = sum(result) / len(result)
    return accuracy


def extract_year(string):
    if string is None:
        return None
    pattern = r"\d{4}"  # Matches any 4-digit number
    match = re.search(pattern, string)
    if match:
        return match.group()
    else:
        return None

def format_example_ground_truth(value, target_attribute, template):
    concept_class = None
    for concept in template[TARGET_ATTRIBUTES]:
        if target_attribute in template[TARGET_ATTRIBUTES][concept]:
            concept_class = concept
            break
    if concept_class == ConceptClass.YEAR.value:
        return str(extract_year(value))
    else:
        return value
def generate_icl_query_indirect(examples, query, target_attribute, template):
    contextualising_attributes = template[CONTEXTUALISING_ATTRIBUTES]
    examples_queries = []
    for idx, row in examples.iterrows():
        examples_query = []
        for attribute in contextualising_attributes:
            if row[attribute] is not None:
                examples_query.append(row[attribute])
        examples_query = ', '.join(examples_query)
        examples_query += (': ' + format_example_ground_truth(row[target_attribute], target_attribute, template))
        examples_queries.append(examples_query)
    examples_queries = '\n'.join(examples_queries)
    examples_queries += '\n'
    target_query = []
    for attribute in contextualising_attributes:
        if query[attribute] is not None:
            target_query.append(query[attribute])
    target_query = ', '.join(target_query) + ': '
    query_final = examples_queries + target_query

    return query_final, query[target_attribute]


def get_icl_examples(entities, template, target_attribute, count=500):
    example_count = count
    icl_prompt_indirect = []
    ground_truth = []
    i = 0
    visited = set()
    pbar = tqdm(total=example_count)
    while i < example_count:
        sampled = sample(entities, 4)
        query = sampled[3][INDEX_COLUMN]
        if query in visited:
            continue
        visited.add(query)
        query, target = generate_icl_query_indirect(pd.DataFrame(sampled[:3]), sampled[3], target_attribute, template)
        icl_prompt_indirect.append(query)
        ground_truth.append(target)
        i += 1
        pbar.update(1)
    pbar.close()

    df = pd.DataFrame()
    df[ICL_PROMPT_COLUMN] = icl_prompt_indirect
    df[GROUND_TRUTH] = ground_truth

    return df