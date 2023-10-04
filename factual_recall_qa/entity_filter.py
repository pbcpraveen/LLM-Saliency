"""
Read in verified entities from knowledge_validation/dataset.
Use the optimized prompts from the dataset folder in this directory.
format should be similar to original pickle, but instead of attribute:value[str/int/etc.], it should be attribute:(value[str/int/etc.],verification[bool])
"""
import argparse
import os
import pickle
from tqdm import tqdm
from utils import *
from common_utils import *
from constants import *
from dotenv import load_dotenv

load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up command-line arguments parser
parser = argparse.ArgumentParser(description="Run experiments with specified conditions.")

# Add arguments
parser.add_argument("--model", type=str, required=True,
                    help="Name of the model to query.")
parser.add_argument("--domain", type=str, required=True,
                    help="Name of domain to test (movie, nobel_laureates, or wiki_bio)")
parser.add_argument("--limit", type=int, default=10000000,
                    help="Number of questions to evaluate. Should be -1 (no limit) if using `--evaluate old`")
parser.add_argument("--skip", type=int, default=0,
                    help="Skip a certain number of questions.")
parser.add_argument("--threads", type=int, default=4,
                    help="Number of threads to use for the evaluation.")
args = parser.parse_args()

# Set experimental conditions
MODEL_NAME = args.model
DOMAIN = args.domain
META = metadata[DOMAIN]
LIMIT = args.limit
SKIP = args.skip
THREADS = args.threads
SCORER = name_similarity

in_path = os.path.abspath(os.path.join('..', 'knowledge_validation', 'dataset', VERIFIED_RECORDS[DOMAIN]))
out_path = os.path.abspath(os.path.join('..', 'knowledge_validation', 'dataset',
                                        VERIFIED_RECORDS[DOMAIN].replace('verified', 'attribute_verified')))

# %%
# read dataset
setup_directories()
logger = get_logger(DOMAIN + ".log")
logger.info(f"Loading {DOMAIN} data")
with open(in_path, 'rb') as dat:
    in_data = pickle.load(dat)

entities = tuple(
    [entity for entity in in_data if entity['verified']])  # only use entities where we verified the identity


# %%
def prompt_generator(target):
    global DOMAIN
    global META
    template_txt = '_'.join([DOMAIN, target, 'candidate_prompts.txt'])
    template_path = os.path.join('dataset', template_txt)
    with open(template_path, 'r') as temp:
        template = temp.read()
    generate = lambda entity: get_entity_prompt(META, entity, template, target)
    return generate


def get_id(entity):
    context = [str(attr) for attr in entity[CONTEXTUALISING_ATTRIBUTES].values()]
    return ''.join(context)


# get list of attribute names
concept_classes = META[TARGET_ATTRIBUTES].keys()
for concept_class in concept_classes:
    for attribute in META[TARGET_ATTRIBUTES][concept_class]:
        logger.info(F'domain: {DOMAIN}, concept class: {concept_class}, attribute: {attribute}')
        # check for previously checked
        try:
            with open(out_path, 'rb') as handle:
                entities = pickle.load(handle)
        except FileNotFoundError:  # no previous checked records
            pass  # don't re-do any entities

        logger.info(f"Verifying first {len(entities)} un-checked entities")

        # fill in the prompts for each entity
        try:
            generate_prompt = prompt_generator(attribute)
        except FileNotFoundError:  # prompt file doesn't exist - skip this attribute
            logger.info(f"No prompt file for domain:{DOMAIN} - attribute:{attribute}")
            continue
        prompts = [generate_prompt(entity) for entity in entities]
        answers = [entity[TARGET_ATTRIBUTES][concept_class][attribute] for entity in entities]
        # print(len(prompts), len(answers))
        # print(prompts[0], '\n', answers[0])
        # continue

        # query the model

        responses = create_and_run_api_request_threads(prompts, THREADS, logger, model=MODEL_NAME)
        response_texts = list(itertools.chain(*responses))
        for j in range(len(response_texts)):
            if 'response' not in entities[j].keys():
                entities[j]['response'] = {}
            entities[j]['response'][attribute] = response_texts[j]

        #for e, p, a, r in zip(entities, prompts, answers, response_texts):  # check repetition scores
        for idx in range(len(entities)):
            if 'attribute_verified' not in entities[idx].keys():
                entities[idx][ATTRIBUTE_VERIFIED] = {}
                entities[idx]['prompt'] = {}
                entities[idx]['response'] = {}
            entities[idx]['prompt'][attribute] = prompts[idx]
            entities[idx]['response'][attribute] = response_texts[idx]
            if answers[idx] is not None:
                entities[idx][ATTRIBUTE_VERIFIED][attribute] = score_attribute(concept_class, response_texts[idx], answers[idx])
            else:
                entities[idx][ATTRIBUTE_VERIFIED][attribute] = False

        logger.info(
            f"Total # entities checked: {len([ent for ent in entities if attribute in ent['attribute_verified']])}")
        logger.info(
            f"Total # entities correctly identified (attribute={attribute}): {len([ent for ent in entities if attribute in ent['attribute_verified'] and ent['attribute_verified'][attribute]])}")

        with open(out_path, 'wb') as handle:
            pickle.dump(entities, handle)
