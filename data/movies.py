from utils import *
from tqdm import tqdm
import pickle
import pandas as pd

import kaggle

kaggle.api.authenticate()

setup_directories()
logger = get_logger('movies.log')
logger.info("checking if dataset is available locally")
kaggle.api.dataset_download_files(MOVIE_DATASET, path='dataset/', unzip=True)


data = pd.read_csv("dataset/imdb_top_1000.csv").to_dict('records')


logger.info("Processing data with {} records".format(len(data)))
structured_data = []
meta = metadata[MOVIE]
pbar = tqdm(total=len(data))
for i in data:
    record = get_record(i, metadata[MOVIE])
    if validate_record(record):
        structured_data.append(record)
    pbar.update(1)

pbar.close()

logger.info("writing data with {} records to file".format(len(structured_data)))
with open('dataset/movies.pickle', 'wb') as handle:
    pickle.dump(structured_data, handle)



