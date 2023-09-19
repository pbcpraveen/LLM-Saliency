import pandas as pd
import wget
from utils import *


setup_directories()
logger = get_logger('nobel_prize.log')

logger.info("checking if dataset is available locally")
df = None
if "country_raw.csv" not in os.listdir("dataset/"):
    logger.info("downloading data from {}".format(COUNTRY_DATASET))
    _ = wget.download(COUNTRY_DATASET, out="dataset/country_raw.csv")
    logger.info("writing data to {}".format("dataset/country_raw.csv"))

df = pd.read_csv("dataset/country_raw.csv")
columns = [
    Attribute.COUNTRY_NAME.value,
    Attribute.YEAR.value,
    Attribute.LEADER_NAME.value,
    Attribute.LEADER_POSITION.value
]
leader_data = df[columns]

prime_ministers = leader_data[leader_data[Attribute.LEADER_POSITION.value] == "Prime Minister"]
presidents = leader_data[leader_data[Attribute.LEADER_POSITION.value] == "Presidents"]

