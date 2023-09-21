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


def get_prompt_generator_prompt(entity, target_attribute):
    prompt = (f"I want to prompt a Large Language Model to recall the {target_attribute} of a {entity[ENTITY]} by "
              f"providing the following attributes of the person ")
    for concept in entity[CONTEXTUALISING_ATTRIBUTES]:
        prompt += f"\n- {concept.replace('_', ' ')}"
    prompt += "\n"
    prompt += (f"Provide 10 best prompts with varying lengths. In your response you must have 3 short prompts, "
               f"3 medium length prompts and 4 long prompts. Further, Your prompts must have all "
               f"{len(entity[CONTEXTUALISING_ATTRIBUTES])} attributes of the {entity[ENTITY]}. You must "
               f"just list the prompts in your response and no other text.")
    return prompt
