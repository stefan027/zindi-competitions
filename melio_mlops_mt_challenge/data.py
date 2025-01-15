"""Data functions"""

import os
import random
import json

import torch
from torch.utils.data import Dataset
from fastai.data.load import DataLoader
from fastai.data.core import DataLoaders
import datasets
import pandas as pd


def ds2df(ds):
    """
    Converts a HuggingFace `datasets.dataset_dict.DatasetDict` to a `pandas.DataFrame`.

    Args:
        ds (`datasets.dataset_dict.DatasetDict`):
            The dataset dictionary containing translation data.

    Returns:
        `pandas.DataFrame`:
            A DataFrame containing the translations with columns for 'dyu', 'fr', and the
            dataset split.
    """
    return pd.DataFrame(
        data=[
            [row["translation"]["dyu"], row["translation"]["fr"], split]
            for split in ds.keys() for row in ds[split]
        ],
        columns=["dyu", "fr", "split"]
    )


def load_from_json(
    train_files:str|list, valid_files:str|list=None, test_files:str|list=None,
    return_format:str="ds"
):
    """
    Loads a dataset from local JSON files into either a `datasets.dataset_dict.DatasetDict`
    or a `pandas.DataFrame`.

    Args:
        train_files (`str` | `list`):
            Path(s) to the training data JSON file(s).
        valid_files (`str` | `list`, optional):
            Path(s) to the validation data JSON file(s). Default is None.
        test_files (`str` | `list`, optional):
            Path(s) to the test data JSON file(s). Default is None.
        return_format (`str`):
            The format to return the data in: 'ds' for a `datasets.dataset_dict.DatasetDict` 
            or 'df' for a `pandas.DataFrame`. Default is 'ds'.

    Returns:
        `datasets.dataset_dict.DatasetDict` or `pandas.DataFrame`:
            The loaded dataset in the specified format.
    """
    ds = datasets.load_dataset(
        "json",
        data_files={
            "train": train_files,
            "validation": valid_files,
            "test": test_files
        },
        field="data",
        features=datasets.Features(
            {
                "ID": datasets.Value("string"),
                "translation": {
                    "dyu": datasets.Value("string"),
                    "fr": datasets.Value("string")}
            }
        )
    )
    if return_format == "ds":
        return ds
    if return_format == "df":
        return ds2df(ds)
    raise ValueError("Invalid return format")


def save_data_locally(ds, save_dir="./data"):
    """
    Saves a dataset locally as JSON files, one for each data split (train, validation, test).

    Args:
        ds (`datasets.dataset_dict.DatasetDict`):
            The dataset dictionary containing the splits to save.
        save_dir (`str`):
            The directory where the dataset files will be saved. Default is "./data".
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for split in ("train", "validation", "test"):
        d = {
            "split": split,
            "data": [{
                "ID": row["ID"], "translation": {
                    "dyu": row["translation"]["dyu"],
                    "fr": row["translation"]["fr"]
                }} for row in ds[split]
            ]
        }
        with open(os.path.join(save_dir, f"dataset_{split}.json"), "w") as f:
            json.dump(d, f, indent=4)


class TranslationDataset(Dataset):
    """
    A PyTorch `Dataset` for the Dyula to French translation task.

    Args:
        df (`pandas.DataFrame`):
            A DataFrame containing the translation data. Each row represents a translation 
            pair with the first column being the source text and the second column 
            being the target translation.
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer to use for encoding the source and target languages.
        bs (`int`):
            Batch size for the dataset. Default is 32.
        is_validation (`bool`):
            Flag indicating whether the dataset is used for validation or training.
            If True, batches are sampled sequentially. Default is False.
        src_lang (`str`):
            The source language code, such as 'dyu_Latn'. Default is 'dyu_Latn'.
        tgt_lang (`str`):
            The target language code, such as 'fra_Latn'. Default is 'fra_Latn'.
        preproc_func (`callable`, optional):
            Preprocessing function to apply to each example before tokenization.
            Default is `None`.
        max_length (`int`):
            Maximum length for tokenized sequences. Default is 128.

    Returns:
        `TranslationDataset`:
            A dataset object used for the Dyula to French translation task.
    """
    def __init__(
        self, df, tokenizer, bs=32, is_validation=False, src_lang="dyu_Latn",
        tgt_lang="fra_Latn", preproc_func=None, max_length=128
    ):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.is_validation = is_validation
        self.n = len(df)
        self.bs = bs
        if preproc_func is None:
            self.preproc_func = lambda x: x
        self.preproc_func = preproc_func
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.max_length = max_length

    def __getitem__(self, i):
        if self.is_validation:
            idx = list(range(i*self.bs, min(i*self.bs+self.bs, self.n), 1))
        else:
            idx = [random.randint(0, self.n-1) for _ in range(self.bs)]
        b = self.df.iloc[idx]
        dyu = b.iloc[:, 0].apply(self.preproc_func).tolist()
        fra = b.iloc[:, 1].apply(self.preproc_func).tolist()

        self.tokenizer.src_lang = self.src_lang
        x = self.tokenizer(
            dyu, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length
        )
        self.tokenizer.src_lang = self.tgt_lang
        y = self.tokenizer(
            fra, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length
        )
        y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = -100
        x["labels"] = y.input_ids
        y["indexes"] = torch.tensor(idx)
        return x, y

    def __len__(self):
        return len(self.df) // self.bs


def create_dataloaders(
    df_train, df_valid, tokenizer, bs=32, src_lang="dyu_Latn", tgt_lang="fra_Latn",
    preproc_func=None, max_length=128, validation_bs=None
):
    """
    A convenience function to create fastai `Dataloaders` for the Dyula to French
    translation task.

    Args:
        df_train (`pandas.DataFrame`):
            The training dataframe containing translation pairs.
        df_valid (`pandas.DataFrame`):
            The validation dataframe containing translation pairs.
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer to use for encoding the source and target languages.
        bs (`int`):
            Batch size for the training dataset. Default is 32.
        src_lang (`str`):
            The source language code, such as 'dyu_Latn'. Default is 'dyu_Latn'.
        tgt_lang (`str`):
            The target language code, such as 'fra_Latn'. Default is 'fra_Latn'.
        preproc_func (`callable`, optional):
            Preprocessing function to apply to each example before tokenization.
              Default is `None`.
        max_length (`int`):
            Maximum length for tokenized sequences. Default is 128.
        validation_bs (`int`, optional):
            Batch size for the validation dataset. Defaults to the training batch size
            if not specified.

    Returns:
        `fastai.DataLoaders`:
            A `DataLoaders` object for training and validation datasets.
    """
    validation_bs = bs if validation_bs is None else validation_bs
    ds_train = TranslationDataset(
        df_train, tokenizer, bs=bs, preproc_func=preproc_func, src_lang=src_lang, tgt_lang=tgt_lang
    )
    ds_valid = TranslationDataset(
        df_valid, tokenizer, bs=validation_bs, is_validation=True, preproc_func=preproc_func,
        src_lang=src_lang, tgt_lang=tgt_lang
    )
    dl_train = DataLoader(ds_train, bs=None)
    dl_valid = DataLoader(ds_valid, bs=None)
    return DataLoaders(dl_train, dl_valid)
