# %%
import os
import pickle
# from utils import *
from tqdm import tqdm
from common_utils import *

openai.organization = os.getenv('OPENAI_API_ORG_LEI')

# movie_path = os.path.abspath('..\data\dataset\movies.pickle')
# verified_movie_path = os.path.abspath('dataset\movies_verified.pickle')
movie_path = os.path.abspath('data\dataset\movies.pickle')
verified_movie_path = os.path.abspath(f'knowledge_validation\dataset\{VERIFIED_RECORDS[MOVIE]}')

verify_limit = 2 # verify all entities
n_threads = 1
model = 'gpt-4-0314'
temp = 0

# %%
# read dataset
setup_directories()
logger = get_logger("movies.log")
logger.info("Loading movies data")
with open(movie_path, 'rb') as mov:
    movies = pickle.load(mov)
# %%
# function for identifying entities
def get_movie_id (entity):
    context = tuple(entity[CONTEXTUALISING_ATTRIBUTES].values())
    return ''.join(context)

# generate prompts
logger.info("Generating movies verification prompts")
entities = {}
prompts = {}
for entity in movies:
    id = get_movie_id(entity) # get the value of the movie id (first 2 contextrualising attributes)
    prompt = f"What is the full name of the {entity['entity']} with the following associated information?:"
    for concept, value in tuple(entity[CONTEXTUALISING_ATTRIBUTES].items())[1:]:
        prompt += f"\n- {concept.replace('_',' ')}: {value}"
    for concept_class in entity[TARGET_ATTRIBUTES]:
        for concept, value in entity[TARGET_ATTRIBUTES][concept_class].items():
            prompt += f"\n- {concept.replace('_',' ')}: {value}"
    prompts[id] = prompt
    entities[id] = entity
# sample movie+prompt, for demonstration
logger.info(f"Entity name: {list(prompts.items())[0][0]}")
logger.info(f"Prompt: {list(prompts.items())[0][1]}")

# %%
# evaluation function
def entity_scorer (generated:str, target:str):
    """
    Scorer function, comparing the generated text to the target.
    Returns True if the generated matches the target, False otherwise
    """
    if target.lower() in generated.lower(): # the generated text may have extra words, but it should contain the target value.
        return True
    return False

# %%
# check previously verified entities
try:
    with open(verified_movie_path, 'rb') as ver:
        verified = pickle.load(ver)
except FileNotFoundError:
    verified = []

verified_ids = [get_movie_id(entity) for entity in verified]
logger.info(f"{len(verified_ids)} entities already verified")

# %%
# query model
missed = []
prompt_list = list(prompts.items())
prompt_list = [(id, prompt) for id, prompt in prompt_list if id not in verified_ids]
if verify_limit:
    prompt_list = prompt_list[:max(verify_limit, len(prompt_list))]
ids, prompts = tuple(zip(*prompt_list))
logger.info(f"Verifying first {len(prompt_list)} un-verified entities")
responses = create_and_run_api_request_threads(prompts, n_threads, logger)
response_texts = ';'.join([';'.join(batch) for batch in responses]) # concatenate all response texts

for id in ids: # check repetition scores
    new_ent = entities[id]
    new_ent['verified'] = entity_scorer(response_texts, id)
    verified.append(entities[id])

print("Total # entities checked:", len(verified))
print("Total # entities correctly identified:", len([ent for ent in verified if ent['verified']]))

with open(verified_movie_path, 'wb') as handle:
    pickle.dump(verified, handle)

# %%



