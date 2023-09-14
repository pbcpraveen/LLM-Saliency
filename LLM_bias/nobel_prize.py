import wget
from constants import *
import os
from utils import *
import json
from tqdm import tqdm
import pickle


logger = get_logger('nobel_prize.log')

logger.info("checking if dataset is available locally")

if "nobel_laureates.json" not in os.listdir('dataset'):
    logger.info("downloading dataset from {}".format(NOBEL_PRIZE_DATASET))
    _ = wget.download(NOBEL_PRIZE_DATASET, out="dataset/nobel_laureates.json", bar=wget_bar_progress)
    logger.info("writing data to {}".format("dataset/nobel_laureates.json"))


with open('dataset/nobel_laureates.json') as json_file:
   json_data = json.load(json_file)

logger.info("Processing data with {} records".format(len(json_data)))
structured_data = []
meta = metadata[NOBEL_PRIZE]
pbar = tqdm(total=len(json_data))
for i in json_data:
    record = get_record(i, metadata[NOBEL_PRIZE])
    if validate_record(record):
        structured_data.append(record)
    pbar.update(1)

pbar.close()

logger.info("writing data with {} records to file".format(len(structured_data)))
with open('dataset/noble_prize.pickle', 'wb') as handle:
    pickle.dump(structured_data, handle)



