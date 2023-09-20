import logging
import threading
import time

from tqdm import tqdm

from constants import *
from logging.handlers import RotatingFileHandler
import sys
import os
import openai
from threading import Thread
from dotenv import load_dotenv

responses = []


load_dotenv('api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

def chatgpt_query(query, model = "gpt-4-0314", temperature=0):
    response = openai.ChatCompletion.create(
            model=model,
            messages=query,
            temperature=temperature,
            request_timeout=90,
            max_tokens=150
            )

    return response.choices[0].message["content"].replace('\n', ' ')


def setup_directories():
    if not os.path.exists('dataset/'):
        os.makedirs('dataset/')
    if not os.path.exists('logs/'):
        os.makedirs('logs/')


def get_logger(log_file, depth=logging.DEBUG):
    logging.basicConfig(filename="logs/{}".format(log_file),
                        filemode='a')

    logger = logging.getLogger()
    logger.setLevel(depth)

    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s')

    handler = RotatingFileHandler("logs/{}".format(log_file), maxBytes=1024*1024*5, backupCount=1)
    handler.setFormatter(log_formatter)
    handler.setLevel(depth)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def query_thread(prompts, global_index):
    global responses
    count = len(prompts)
    i = 0
    responses_thread = []
    pbar = tqdm(total=count)
    while len(responses_thread) != count:
        try:
            query = [
                {"role": "user", "content": prompts[i]}
            ]
            response = chatgpt_query(query)
            i += 1
            responses_thread.append(response)
            pbar.update(1)
        except Exception as e:
            time.sleep(10)
    pbar.close()
    responses[global_index] = responses_thread


def create_and_run_api_request_threads(queries, n_threads, logger, temperature=0):
    global responses

    count = len(queries)
    responses = [[] for _ in range(n_threads)]
    partitions = []
    bin_size = count // n_threads

    for i in range(n_threads - 1):
        partitions.append(queries[i * bin_size: (i + 1) * bin_size])

    partitions.append(queries[(n_threads - 1) * bin_size:])

    threads = []
    for i in range(n_threads):
        threads.append(threading.Thread(target=query_thread, args=(partitions[i], i,)))

    logger.info("starting API resquests to OPENAI's GPT 4 using " + str(n_threads) + " threads")
    logger.info("Number of threads created: " + str(len(threads)))
    logger.info("Number of partitions created: " + str(len(partitions)))
    logger.info("Size of each partition: " + str(bin_size))

    for i in range(n_threads):
        threads[i].start()
    for i in range(n_threads):
        threads[i].join()

    return responses
