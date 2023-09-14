## Setup


To download and set up the data with specified schema, follow the below steps

- Define the `EntityClass`, `ConceptClasses` and `Attributes` in the `data/constants.py` script.
- Delineate the metadata by adding an entry to `metadata` dictionary in the `data/constants` script.
- Write a new python script in `data/` directory and write code to perform the below tasks

    - Get the data in any form a credible source through API calls.
    - Flatten the records in the data to get a list of 1-depth dictionaries, where each dictionary is a record.
    - Now call the `get_record()` function for each record to construct a record with schema mentioned in `constants.metadata`.
    - Validate the records and remove any null values and dump the list of records in to a pickle file in `data/dataset/` directory.

**Note:** For the movie dataset the code need to make a call with kaggle API. Therefore, you need to download the kaggle API key (`.kaggle`)
from this [link](https://www.kaggle.com/settings/account) - navigate to API section and generate new token. Now copy this token file to
`/users/<username>/.kaggle/` directory.
