"""Basic utility and helper functions"""
import gc
import random
import re
import torch
import numpy as np
import pandas as pd
import datasets


CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]]'


def cleanup():
    """
    Free GPU memory by performing garbage collection and clearing the CUDA cache.
    """
    gc.collect()
    torch.cuda.empty_cache()


def set_seed(s, reproducible=False):
    """
    This function is from the fastai library. Set random seed for `random`, `torch`,
    and `numpy` (where available)
    """
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s%(2**32-1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preproc(text: str) -> str:
    """
    This function is copied from the competition's examples repo.
    Clean input text by removing special characters and converting
    to lower case.
    """
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
    return text.strip()


def get_example(
    examples, idx:int=None, return_reference=False, src="dyu", tgt="fr"
):
    """
    Retrieve an example from the dataset or DataFrame.

    Args:
        examples (datasets.arrow_dataset.Dataset | pd.DataFrame): 
            The dataset or DataFrame from which to retrieve the example.
        idx (int, optional): 
            The index of the example to retrieve. If not provided, a random index is chosen.
        return_reference (bool, optional): 
            If True, returns both the source and target translations.
            If False, only returns the source translation.
        src (str, optional): 
            The source language key in the example. Default is 'dyu'.
        tgt (str, optional): 
            The target language key in the example. Default is 'fr'.

    Returns:
        Union[str, Tuple[str, str]]:
            The source translation if `return_reference` is False. Otherwise, a tuple containing
            the source and target translations.
    """
    if idx is None:
        idx = random.randint(0, len(examples)-1)
    if isinstance(examples, datasets.arrow_dataset.Dataset):
        example = examples[idx]["translation"]
    elif isinstance(examples, pd.DataFrame):
        example = examples.iloc[idx]
    else:
        raise TypeError
    if return_reference:
        return example[src], example[tgt]
    return example[src]
