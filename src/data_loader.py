import pandas as pd
import re
import string
import logging
from datasets import Dataset

class Normalizer:
    """
    Cleans and normalizes English and Persian text.
    """
    def __init__(self):
        translation_src = ' ىكي“”0123456789%إأآئيؤةك'
        translation_dst = ' یکی""۰۱۲۳۴۵۶۷۸۹٪اااییوهک'
        self.translations = str.maketrans(translation_src, translation_dst)

        patterns = [
            (r' {2,}', ' '),      # Remove extra spaces
            (r'\n+', ' '),        # Replace newlines with space
            (r'\u200c+', ' '),    # Replace ZWNJs with space
            (r'[ـ\r]', '')        # Remove keshide, carriage returns
        ]
        self.character_refinement_patterns = [(re.compile(pattern), repl) for pattern, repl in patterns]

    def normalize_fa(self, text):
        if not isinstance(text, str): return ""
        text = text.lower().translate(self.translations)
        text = re.sub('[^a-zA-Z۰-۹آ-ی ]', ' ', text)
        for pattern, repl in self.character_refinement_patterns:
            text = pattern.sub(repl, text)
        return text.strip()

    def normalize_en(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
        return text.strip()

def apply_preprocess(datasets, configs):
    """
    Applies normalization to the dataset rows.
    """
    normalizer = Normalizer()
    fa_col = configs['persian']
    en_col = configs['english']

    def apply_row_normalization(example):
        example[fa_col] = normalizer.normalize_fa(example[fa_col])
        example[en_col] = normalizer.normalize_en(example[en_col])
        return example

    new_datasets = []
    for dataset in datasets:
        new_datasets.append(dataset.map(apply_row_normalization, num_proc=4))
    return new_datasets

def get_dataloaders(configs):
    """
    Loads and preprocesses datasets from CSV files.
    """
    logger = logging.getLogger(__name__)
    try:
        df_train = pd.read_csv(configs['train_path'])
        df_val = pd.read_csv(configs['val_path'])
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}. Please download train.csv and val.csv.")
        raise

    if df_train.empty or df_val.empty:
        logger.error("One or both datasets are empty.")
        raise ValueError("Datasets cannot be empty.")

    # Rename columns to standard names
    df_train = df_train.rename(columns={"en": configs['english'], "fa": configs['persian']})
    df_val = df_val.rename(columns={"en": configs['english'], "fa": configs['persian']})

    dataset_train = Dataset.from_pandas(df_train[[configs['english'], configs['persian']]])
    dataset_val = Dataset.from_pandas(df_val[[configs['english'], configs['persian']]])
    
    logger.info("Applying text normalization...")
    dataset_train, dataset_val = apply_preprocess([dataset_train, dataset_val], configs)
    
    logger.info(f"Sample data after preprocessing: {dataset_train[0]}")
    
    return dataset_train, dataset_val
