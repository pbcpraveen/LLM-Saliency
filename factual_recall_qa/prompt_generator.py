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
import random
import time

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import propmts
from common_utils import *
from constants import *
from utils import *

load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

setup_directories()
logger = get_logger(F"{NOBEL_PRIZE}.log", depth="INFO")

meta = metadata[NOBEL_PRIZE]
verified_record = pickle.load(open(f"../knowledge_validation/dataset/{VERIFIED_RECORDS[NOBEL_PRIZE]}", "rb"))

target_attribute = Attribute.DEATH_CITY.value
display_target_attribute = "city of death"
concept_class = None

for i in meta[TARGET_ATTRIBUTES].keys():
    if target_attribute in meta[TARGET_ATTRIBUTES][i]:
        concept_class = i

# filter records which has value for the target attribute.
filtered_records = []
for i in range(len(verified_record)):
    if verified_record[i][TARGET_ATTRIBUTES][concept_class][target_attribute] is not None:
        filtered_records.append(verified_record[i])

#Querying LLM for candidate Prompts
generator_prompt = get_prompt_generator_prompt(meta, display_target_attribute)
query = [{"role": "user", "content": generator_prompt}]
response = chatgpt_query(query, replace_newline=True, max_tokens=300)

candidate_prompts = list(map(remove_numeric_bullets, response.split("$")))
logger.info("Candidate Prompts Generated:")
for i in candidate_prompts:
    logger.info(i)

test_sample = random.sample(filtered_records, 50)
responses = []

logger.info("Preparing test prompts to evaluate the candidate templates.")
test_prompts = {}
for prompt in candidate_prompts:
    test_prompts[prompt] = []
    for sample in test_sample:
        prompt_generated = get_entity_prompt(meta, sample, prompt, display_target_attribute)
        test_prompts[prompt].extend([(prompt_generated, sample[TARGET_ATTRIBUTES][concept_class][target_attribute])] * 5)

df = pd.DataFrame()
df[PROMPT_INDEX_COLUMN] = list(itertools.chain(*[[i] * 250 for i in range(len(candidate_prompts))]))
input_output_pair = list(itertools.chain(*[test_prompts[prompt] for prompt in test_prompts.keys()]))
df[PROMPT_COLUMN] = [pair[0] for pair in input_output_pair]
df[GROUND_TRUTH] = [pair[1] for pair in input_output_pair]

logger.info("Running API requests for the test prompts.")
query_prompts = df[PROMPT_COLUMN].to_list()
responses = create_and_run_api_request_threads(query_prompts, 5, logger, temperature=1)
responses = list(itertools.chain(*responses))

df[GPT_4_RESPONSE] = responses

logger.info(f"Writing GPT4 response to dataset/{meta[ENTITY]}_candidate_prompts.csv")
df.to_csv(f'dataset/{meta[ENTITY]}_{target_attribute}_candidate_prompts.csv')
time.sleep(10)


prompt_index = df[PROMPT_INDEX_COLUMN].unique().tolist()

d = {}
prompts = df['prompt'].unique().tolist()
for index in prompt_index:
    d[index] = get_score(df, index, concept_class=concept_class)
max_score = max([i[1] for i in d.items()])

candidates = []
for i in d.items():
    if i[1] == max_score:
        candidates.append(i[0])

prompts_selected = [candidate_prompts[i] for i in candidates]
best_prompt = max(prompts_selected, key=len)
with open(f"dataset/{meta[ENTITY]}_{target_attribute}_candidate_prompts.txt", "w") as text_file:
    text_file.write(best_prompt)
time.sleep(10)