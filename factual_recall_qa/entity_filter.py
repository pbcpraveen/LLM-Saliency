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
openai.organization = os.getenv('OPENAI_API_ORG_LEI')



# Set up command-line arguments parser
parser = argparse.ArgumentParser(description="Run experiments with specified conditions.")

# Add arguments
parser.add_argument("--model", type=str, required=True,
                    help="Name of the model to query.")
parser.add_argument("--domain", type=str, required=True,
                    help="Name of domain to test (movie, nobel_laureates, or wiki_bio)")
parser.add_argument("--limit", type=int, default=-1,
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

in_path = os.path.abspath(os.path.join('..','knowledge_validation','dataset',VERIFIED_RECORDS[DOMAIN]))
out_path = os.path.abspath(os.path.join('..','knowledge_validation','dataset',VERIFIED_RECORDS[DOMAIN].replace('verified', 'attribute_verified')))

# %%
# read dataset
setup_directories()
logger = get_logger(DOMAIN+".log")
logger.info(f"Loading {DOMAIN} data")
with open(in_path, 'rb') as dat:
    in_data = pickle.load(dat)

entities = tuple([entity for entity in in_data if entity['verified']]) # only use entities where we verified the identity
# %%
def prompt_generator (target):
        template_txt = '_'.join([DOMAIN,target, 'candidate_prompts.txt'])
        template_path = os.path.join('dataset',template_txt)
        with open(template_path, 'r') as temp:
            template = temp.read()
        def generate(entity):
            return get_entity_prompt(META, entity, template, target)
        return generate
def get_id (entity):
    context = [str(attr) for attr in entity[CONTEXTUALISING_ATTRIBUTES].values()]
    return ''.join(context)

# get list of attribute names
concept_classes = META[TARGET_ATTRIBUTES].keys()
for concept_class in concept_classes:
    for attribute in META[TARGET_ATTRIBUTES][concept_class]:
        logger.info(F'domain: {DOMAIN}, concept class: {concept_class}, attribute: {attribute}')
        # check for previously checked
        checked = []
        try:
            with open(out_path, 'rb') as handle:
                checked = pickle.load(handle)
        except FileNotFoundError: # no previous checked records
            pass
        checked = {get_id(entity):entity for entity in checked if attribute in entity['attribute_verified']}
        new_entities = [entity for entity in entities if get_id(entity) not in checked.keys()] # don't re-do any entities
        if LIMIT:
            new_entities = new_entities[:min(LIMIT, len(new_entities))]
        logger.info(f"Verifying first {len(new_entities)} un-checked entities")

        # fill in the prompts for each entity
        try:
            generate_prompt = prompt_generator(attribute)
        except FileNotFoundError: # prompt file doesn't exist - skip this attribute
            logger.info(f"No prompt file for domain:{DOMAIN} - attribute:{attribute}")
            continue
        prompts = [generate_prompt(entity) for entity in new_entities]
        answers = [entity[TARGET_ATTRIBUTES][concept_class][attribute] for entity in new_entities]
        # print(len(prompts), len(answers))
        # print(prompts[0], '\n', answers[0])
        # continue

        # query the model
        responses = create_and_run_api_request_threads(prompts, THREADS, logger, model=MODEL_NAME)
        response_texts = [response for batch in responses for response in batch] # flatten the responses list

        for e, p, a, r in zip(new_entities, prompts, answers, response_texts): # check repetition scores
            if 'attribute_verified' not in e:
                e['attribute_verified'] = {}
            e['attribute_verified'][attribute] = score_attribute(concept_class, r, a)
            checked[get_id(e)] = e

        checked = list(checked.values())
        logger.info(f"Total # entities checked: {len([ent for ent in checked if attribute in ent['attribute_verified']])}")
        logger.info(f"Total # entities correctly identified (attribute={attribute}): {len([ent for ent in checked if attribute in ent['attribute_verified'] and ent['attribute_verified'][attribute]])}")

        with open(out_path, 'wb') as handle:
            pickle.dump(checked, handle)
