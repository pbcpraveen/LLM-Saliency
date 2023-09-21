# %%
import os
import pandas as pd
import pickle
from utils import *
from tqdm import tqdm
from common_utils import *

movie_path = '../data/dataset/movies.pickle'
verified_movie_path = '../data/dataset/movies_verified.pickle'
verify_limit = -1 # verify all entities

model = 'gpt-4-0314'
temp = 0

# %%
# read dataset
with open(movie_path, 'rb') as mov:
    movies = pickle.load(mov)
# %%
# generate prompts
entities = {}
prompts = {}
for entity in movies:
    title = tuple(entity[CONTEXTUALISING_ATTRIBUTES].items())[0][1] # get the value of the first contextrualising attribute (movie title)
    prompt = f"What is the full name of the {entity['entity']} with the following associated information?:"
    for concept, value in tuple(entity[CONTEXTUALISING_ATTRIBUTES].items())[1:]:
        prompt += f"\n- {concept.replace('_',' ')}: {value}"
    for concept_class in entity[TARGET_ATTRIBUTES]:
        for concept, value in entity[TARGET_ATTRIBUTES][concept_class].items():
            prompt += f"\n- {concept.replace('_',' ')}: {value}"
    prompts[title] = prompt
    entities[title] = entity
# sample movie+prompt, for demonstration
# print("Entity name:", list(prompts.items())[0][0])
# print("Prompt:", list(prompts.items())[0][1])

# %%
# evaluation function
def entity_scorer (generated:str, target:str):
    """
    Scorer function, comparing the generated text to the target.
    Returns True if the generated matches the target, False otherwise
    """
    if target.lower() in generated.lower():
        return True
    return False

# %%
# check previously verified entities
try:
    with open(verified_movie_path, 'rb') as ver:
        verified = pickle.load(ver)
except FileNotFoundError:
    verified = []

verified_titles = [tuple(entity[CONTEXTUALISING_ATTRIBUTES].items())[0][1] for entity in verified]

# %%
# query model - this does not use the logger at all

missed = []
prompt_list = list(prompts.items())[:verify_limit]
pbar = tqdm(total=len(prompt_list))
for title, prompt in prompt_list:
    if title in verified_titles: # do not repeat entities
        print("Already verified:", title)
        pbar.update(1)
        continue
    messages = [{'role': 'user', 'content': prompt}]
    response = chatgpt_query(messages, model=model, temperature=temp)
    if entity_scorer(response, title):
        verified.append(entities[title])
        verified_titles.append(title)
    else:
        missed.append(entities[title])
    pbar.update(1)
pbar.close()
print("# entities correctly identified:", len(verified), "out of", len(prompt_list))

with open(f"dataset/{VERIFIED_RECORDS[MOVIE]}", 'wb') as handle:
    pickle.dump(verified, handle)

# %%



