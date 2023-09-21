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
from utils import *

load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

setup_directories()
logger = get_logger(F"{NOBEL_PRIZE}.log")

meta = metadata[NOBEL_PRIZE]
verified_record = pickle.load(open(f"../knowledge_validation/dataset/{VERIFIED_RECORDS[NOBEL_PRIZE]}", "rb"))
target_attribute = Attribute.YEAR.value
concept_class = None

for i in meta[TARGET_ATTRIBUTES].keys():
    if target_attribute in meta[TARGET_ATTRIBUTES][i]:
        concept_class = i

# filter records which has value for the target attribute.
filtered_records = []
for i in range(len(verified_record)):
    if verified_record[i][TARGET_ATTRIBUTES][concept_class][target_attribute] is not None:
        filtered_records.append(verified_record[i])

generator_prompt = get_prompt_generator_prompt(meta, target_attribute)
query = [{"role": "user", "content": generator_prompt}]
response = chatgpt_query(query, replace_newline=False, max_tokens=300)
print(response.split('\n'))


