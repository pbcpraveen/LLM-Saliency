import logging
from constants import *
from logging.handlers import RotatingFileHandler
import sys
import os


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


def format_name_capital(name):
    formatted_name = ""

    # Split the name into individual words
    words = name.split()

    for i, word in enumerate(words):
        # Define a list of words to preserve in lowercase
        lowercase_words = ["de", "van", "von", "di", "del", "da", "el", "la", "le", "i", "of"]

        # Check if the word should be preserved in lowercase
        if word.lower() in lowercase_words and i != 0:
            formatted_name += word.lower()  # Preserve the word in lowercase
        else:
            if word == ".":
                continue
            if "-" in word:
                word = "-".join(list(map(lambda x: x.capitalize(), word.split("-"))))
                formatted_name += word
            elif word[:2] == "mc":
                formatted_name += ("Mc" + word[2:].capitalize())
            else:
                formatted_name += word.capitalize()  # Capitalize other words

        # Add a space between words
        if i != len(words) - 1:
            formatted_name += " "

    return formatted_name.strip()


def format_name(input_name):
    input_name = input_name.replace("\n", "")
    if " ," in input_name:
        input_name = input_name.replace(" ,", ",")
    # Check if the input contains '-lrb-' and '-rrb-'
    if "-lrb-" in input_name and "-rrb-" in input_name:
        # Replace '-lrb-' and '-rrb-' with '(' and ')'
        input_name = input_name.replace("-lrb-", "(").replace("-rrb-", ")")
        input_name = input_name.replace("( ", "(").replace(" )", ")")
        formatted_name = input_name.split("(")

        final = " (".join([format_name_capital(formatted_name[0]), formatted_name[1].lower()])
    # Capitalize the first letter of each word
    else:
        final = format_name_capital(input_name)

    return final


def get_record(raw_record, meta):
    record = {
                ENTITY: EntityClass.PERSON.value,
                CONTEXTUALISING_ATTRIBUTES: {},
                TARGET_ATTRIBUTES: {}
    }
    c_attr = {}
    for i in meta[CONTEXTUALISING_ATTRIBUTES]:
        c_attr[i] = raw_record.get(i)
    record[CONTEXTUALISING_ATTRIBUTES] = c_attr

    t_attr = {}
    for concept_class in meta[TARGET_ATTRIBUTES].keys():
        con_attr = {}
        for j in meta[TARGET_ATTRIBUTES][concept_class]:
            con_attr[j] = raw_record.get(j)
        t_attr[concept_class] = con_attr
    record[TARGET_ATTRIBUTES] = t_attr

    return record


def validate_record(record):
    flag = not (None in list(record[CONTEXTUALISING_ATTRIBUTES].values()))
    for i in record[TARGET_ATTRIBUTES].keys():
        flag = (not (None in list(record[TARGET_ATTRIBUTES][i].values()))) and flag
    return flag


def wget_bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()