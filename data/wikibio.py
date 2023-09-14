from tqdm import tqdm
from datasets import load_dataset
from utils import *
import pickle
from constants import metadata

logger = get_logger("wikibio.log")

logger.info("Loading wikibio data")
wikibio_dataset_train = load_dataset(WIKIBIO, split="train")
wikibio_dataset_test = load_dataset(WIKIBIO, split="test")
wikibio_dataset_val = load_dataset(WIKIBIO, split="val")

data = list(wikibio_dataset_train['input_text'])
data.extend(list(wikibio_dataset_test["input_text"]))
data.extend(list(wikibio_dataset_val["input_text"]))


logger.info("Processing data with {} records".format(len(data)))
structured_data = []
meta = metadata[WIKIBIO]
pbar = tqdm(total=len(data))
for i in data:
    record = {}
    fields = i['table']['column_header']
    for j in range(len(fields)):
        record[fields[j]] = i['table']['content'][j]

    record = get_record(record, metadata[WIKIBIO])
    if validate_record(record):
        structured_data.append(record)
    pbar.update(1)

pbar.close()

logger.info("writing data with {} records to file".format(len(structured_data)))
with open('dataset/wikibio_raw.pickle', 'wb') as handle:
    pickle.dump(structured_data, handle)




