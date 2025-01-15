"""Functions to modify and train NllbTokenizer"""

import os
from pathlib import Path
from collections import Counter
import json

import pandas as pd
from transformers import NllbTokenizer, PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_model

from utils import preproc


def limit_tokenizer_vocab(model_id, df, save_dir, min_token_freq=1):
    """
    Limits the vocabulary of a tokenizer based on token frequency in the dataset and
    saves the updated tokenizer.

    This function is used to reduce the size of a tokenizer by retaining only the most
    frequent tokens from the given dataset. Tokens with a frequency below `min_token_freq`
    are removed, and the remaining tokens are saved into a new tokenizer directory along
    with their updated configuration. Special tokens and unknown tokens are handled
    appropriately, and the vocabulary is updated in the sentencepiece model.

    Args:
        model_id (`str`):
            Path or identifier for the pretrained tokenizer model (assumed to be a
            SentencePiece tokenizer).
        df (`pandas.DataFrame`):
            DataFrame containing the dataset with two columns representing the source
            and target language translations. The columns should be named `"dyu"` for
            Dyula and `"fr"` for French.
        save_dir (`str`):
            Directory where the new tokenizer will be saved.
        min_token_freq (`int`, optional):
            The minimum frequency a token must have in the dataset to be included in the
            final tokenizer's vocabulary. Default is 1, meaning all tokens appearing at
            least once are retained.

    Process Overview:
        1. Loads the pretrained tokenizer from `model_id`.
        2. Counts the frequency of each token appearing in the dataset.
        3. Retains tokens that meet the frequency threshold defined by `min_token_freq`.
        4. Checks for frequently occurring tokens that are not in the tokenizer's vocabulary
           and adds them.
        5. Saves the modified tokenizer's configuration, vocabulary, and sentencepiece model
           to `save_dir`.
        6. Ensures that the final tokenizer can be loaded and used with the `transformers`
           library.

    Steps in Detail:
        - The original tokenizer is loaded and applied to the dataset to generate token IDs.
        - Token frequencies are computed for both source (`dyu_Latn`) and target (`fra_Latn`)
          languages.
        - Tokens are filtered based on the minimum frequency and unknown tokens are flagged.
        - The tokenizer's configuration files are modified to reflect the new vocabulary
          and saved.
        - The sentencepiece model is updated by removing unused tokens and adding new tokens.
        - The new tokenizer is tested to ensure compatibility with `transformers` and is saved.

    Notes:
        - Special tokens are preserved and non-relevant special tokens are removed.
        - The tokenizer's vocabulary is updated to include high-frequency tokens that were
          previously unknown.
        - The function handles both updating the tokenizer's configuration files and the
          sentencepiece model.
    """
    # Load existing tokenizer
    tokenizer_old = NllbTokenizer.from_pretrained(model_id)

    # Get a count of how frequently each token appears in the dataset
    token_counts = Counter()
    df_counts = df.copy()
    tokenizer_old.src_lang = "dyu_Latn"
    df_counts["tokens_dyu"] = df_counts["dyu"].apply(
        lambda s: tokenizer_old(preproc(s)).input_ids
    )
    tokenizer_old.src_lang = "fra_Latn"
    df_counts["tokens_fra"] = df_counts["fr"].apply(
        lambda s: tokenizer_old(preproc(s)).input_ids
    )
    df_counts["tokens"] = df_counts["tokens_dyu"] + df_counts["tokens_fra"]
    for tokens in df_counts["tokens"]:
        token_counts.update(dict(Counter(tokens)))
    token_counts = dict(token_counts)
    token_ids = sorted(list(token_counts.keys()))

    # Covert to a Pandas `DataFrame`:
    token_counts = pd.DataFrame(
        {"token": token_counts.keys(), "count": token_counts.values()}
    ).sort_values(by="count", ascending=False)

    # Limit the vocabulary to tokens with at least `min_token_freq` occurences:
    token_ids = sorted(token_counts[token_counts["count"] >= min_token_freq]["token"].tolist())
    print("Number of tokens used:", len(token_ids))

    # We also want to check if there are frequent tokens that are not
    # in the tokenizer's vocabulary.
    token_counts = Counter()
    df_counts = df.copy()
    tokenizer_old.src_lang = "dyu_Latn"
    df_counts["tokens_dyu"] = df_counts["dyu"].apply(
        lambda s: tokenizer_old.tokenize(preproc(s))
    )
    tokenizer_old.src_lang = "fra_Latn"
    df_counts["tokens_fra"] = df_counts["fr"].apply(
        lambda s: tokenizer_old.tokenize(preproc(s))
    )
    df_counts["tokens"] = df_counts["tokens_dyu"] + df_counts["tokens_fra"]
    for tokens in df_counts["tokens"]:
        token_counts.update(dict(Counter(tokens)))
    token_counts = dict(token_counts)

    def _is_unk(t):
        return tokenizer_old.convert_tokens_to_ids(t) == tokenizer_old.unk_token_id

    # Flag all unknown tokens
    token_counts = pd.DataFrame(
        {"token": token_counts.keys(), "count": token_counts.values()}
    ).sort_values(by="count", ascending=False)
    token_counts["is_unk"] = token_counts["token"].apply(_is_unk)

    # The top token occurs very frequently. We'll add that to the tokenizer vocabulary.
    new_tokens = set(token_counts[token_counts["is_unk"]].iloc[0, 0])

    # Save the old tokenizer's configuration files. We will then load the json files
    # and update them as required for our new tokenizer.
    tokenizer_old.save_pretrained("/tmp")

    # Special tokens - keep only relevant language tags
    with open("/tmp/special_tokens_map.json", "r") as f:
        special_tokens_map = json.load(f)
    add_special_tokens = ["dyu_Latn", "fra_Latn"]
    add_special_tokens_remove = set([
        t for t in special_tokens_map["additional_special_tokens"] if t not in add_special_tokens
    ])
    special_tokens_map["additional_special_tokens"] = add_special_tokens

    # Added tokens
    with open("/tmp/added_tokens.json", "r") as f:
        added_tokens = json.load(f)
    added_tokens = {k: v for k, v in added_tokens.items() if k not in add_special_tokens_remove}

    # Tokenizer config
    with open("/tmp/tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)

    added_tokens_decoder = {}
    for k, v in tokenizer_config["added_tokens_decoder"].items():
        if v["content"] not in add_special_tokens_remove:
            added_tokens_decoder[k] = v
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder
    tokenizer_config["additional_special_tokens"] = add_special_tokens

    # Create a folder to save the new tokenizer and save the modified config files
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_tokenizer_dir = Path(save_dir)

    with open(new_tokenizer_dir/"tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f,  indent=4)

    with open(new_tokenizer_dir/"special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f,  indent=4)

    with open(new_tokenizer_dir/"added_tokens.json", "w") as f:
        json.dump(added_tokens, f,  indent=4)

    # The new vocabulary consists of:
    #  - Tokens present in the tokenizer's vocabulary and in our dataset
    #  - Tokens with high frequency in our dataset that are not in the tokenizer's vocab
    #  - The tokenizer's added and special tokens
    new_vocab = (
        {tokenizer_old.convert_ids_to_tokens(i) for i in token_ids}
        .union(
            {v["content"] for v in tokenizer_config["added_tokens_decoder"].values()}
        )
        .union(
            new_tokens
        )
    )
    print("New vocabulary size:", len(new_vocab))

    # Update the `sentencepiece` model
    m = sp_model.ModelProto()
    m.ParseFromString(open("/tmp/sentencepiece.bpe.model", "rb").read())

    # Iterate over `m.pieces` and keep only keep the tokens that are in the new vocab:
    seen = set()
    while True:
        if m.pieces[0].piece in seen:
            break
        x = m.pieces.pop(0)
        seen.add(x.piece)
        if x.piece in new_vocab:
            m.pieces.append(x)

    # Add new tokens:
    add_tokens = new_vocab - seen
    for token in add_tokens:
        new_token = sp_model.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        m.pieces.append(new_token)
    assert len(m.pieces) == len(new_vocab)

    # Serialize the sentencepiece model
    with open(new_tokenizer_dir/'sentencepiece.bpe.model', 'wb') as f:
        f.write(m.SerializeToString())

    # Test loading the updated model:
    sp = spm.SentencePieceProcessor()
    assert sp.load(str(new_tokenizer_dir/'sentencepiece.bpe.model'))

    # Test loading the new tokenizer with `transformers`:
    try:
        tokenizer = NllbTokenizer.from_pretrained(new_tokenizer_dir)
    except Exception:
        print("Cannot load tokenizer with `transformers`")

    # We still need to update the indices in the config files for special tokens with indexes
    added_tokens_new = {}
    for i in range(tokenizer.vocab_size):
        t = tokenizer.convert_ids_to_tokens(i)
        if t in added_tokens:
            print(f"Updated index: {i}: {t}")
            added_tokens_new[t] = i

    added_tokens_decoder_new = {}
    for k, v in added_tokens_decoder.items():
        t = v["content"]
        i = int(k)
        assert t == tokenizer.convert_ids_to_tokens(i)
        if i >= tokenizer.vocab_size:
            i = added_tokens_new[t]
            print(f"Updated index: {i}: {t}")
        added_tokens_decoder_new[str(i)] = v
    tokenizer_config["added_tokens_decoder"] = added_tokens_decoder_new

    # Save the updated configuration files
    with open(new_tokenizer_dir/"tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f,  indent=4)

    with open(new_tokenizer_dir/"added_tokens.json", "w") as f:
        json.dump(added_tokens_new, f,  indent=4)

    # Test loading the final tokenizer with `transformers`:
    try:
        tokenizer = NllbTokenizer.from_pretrained(new_tokenizer_dir)
    except Exception:
        print("Cannot load tokenizer with `transformers`")


def create_tokenizer_train_data(df, fp="data/tokenizer_train_data.txt"):
    """
    Prepares and saves training data for tokenizer training from a given dataset.

    Args:
        df (`pandas.DataFrame`):
            DataFrame containing the dataset with two columns representing the source and
            target language translations. The columns should be named `"dyu"` for Dyula
            and `"fr"` for French.
        fp (`str`, optional):
            File path where the preprocessed training data will be saved. Default is
            `"data/tokenizer_train_data.txt"`.
    """
    base_path = os.path.split(fp)[0]
    if base_path and not os.path.exists(base_path):
        os.makedirs(base_path)

    with open(fp, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{preproc(row['dyu'])}\n")
            if row['fr'] == "0":
                continue
            f.write(f"{preproc(row['fr'])}\n")


def train_new_tokenizer(model_id, df, save_dir, vocab_size=2000):
    """
    Trains a new Sentencepiece tokenizer based on a given dataset and saves it for use
    with a model.

    This function uses a provided dataset to train a new tokenizer. It updates the 
    tokenizer's configuration and special tokens based on the new vocabulary and saves 
    the tokenizer files to the specified directory. 

    Args:
        model_id (`str`):
            The identifier or path to the pre-trained model. This is used to load the 
            existing tokenizer's configuration files and special tokens.
        df (`pandas.DataFrame`):
            DataFrame containing the dataset with at least two columns named `"dyu"` 
            and `"fr"`, representing the source (Dyula) and target (French) translations,
            respectively.
        save_dir (`str`):
            Directory where the newly trained tokenizer and its configuration files will
            be saved.
        vocab_size (`int`, optional):
            Desired size of the tokenizer vocabulary. Default is `2000`.

    Process Overview:
        1. Create training data for the tokenizer from the dataset and save it to a text file.
        2. Load and save the existing tokenizer's configuration files from the specified model.
        3. Train a new tokenizer using the `SentencePieceBPETokenizer` class on the saved
           text data.
        4. Convert the trained tokenizer to a `PreTrainedTokenizerFast` object.
        5. Update the tokenizer configuration files with the new vocabulary and special tokens.
        6. Save the updated tokenizer configuration, special tokens, and additional tokens.
        7. Update the tokenizer's `sentencepiece` model with the new vocabulary and save it.
        8. Test the new tokenizer to ensure it loads correctly and is compatible with
           `transformers`.

    Notes:
        - The `create_tokenizer_train_data` function is used to prepare the dataset for
          tokenizer training.
        - The existing tokenizer's configuration is used to ensure compatibility with the
          new tokenizer.
        - Special tokens and added tokens are preserved and updated in the new tokenizer.
        - The `SentencePieceBPETokenizer` class is used for training the tokenizer. 
        - The `SentencePiece` model is updated and serialized to ensure that the new
          vocabulary is included.
    """
    # Create training data from the `df`
    create_tokenizer_train_data(df)

    # We load the existing tokenizer for the model and save it locally. We'll only
    # use this to get the structure and format of the config files right.
    tokenizer_old = NllbTokenizer.from_pretrained(model_id)
    tokenizer_old.save_pretrained("/tmp")

    # Read the `tokenizer_config.json` file
    with open("/tmp/tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)

    # Read the `special_tokens_map.json` file
    with open("/tmp/special_tokens_map.json", "r") as f:
        special_tokens_map = json.load(f)

    # Read the `added_tokens.json` file
    with open("/tmp/added_tokens.json", "r") as f:
        added_tokens = json.load(f)

    # Train tokenizer
    # Specify the special tokens that we need for our tokenizer
    special_tokens = ['<s>', '<pad>', '</s>', '<unk>', 'dyu_Latn', 'fra_Latn', '<mask>']
    add_special_tokens = ['dyu_Latn', 'fra_Latn']

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(
        "data/tokenizer_train_data.txt",
        vocab_size=vocab_size,
        # min_frequency=5,
        show_progress=True,
        # limit_alphabet=500,
        special_tokens=special_tokens
    )

    # Convert to a Huggingface `PreTrainedTokenizerFast` object:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, clean_up_tokenization_spaces=False
    )

    # Update tokenizer config files
    # Update the `tokenizer_config` that was loaded from the `tokenizer_config.json` file:
    def _added_tokens_decoder_to_dict(token):
        return {
            'content': token,
            'lstrip': False,
            'normalized': False,
            'rstrip': False,
            'single_word': False,
            'special': True
        }
    tokenizer_config["added_tokens_decoder"] = {
        str(i): _added_tokens_decoder_to_dict(t) for t, i in tokenizer.vocab.items()
        if t in special_tokens
    }
    tokenizer_config["additional_special_tokens"] = add_special_tokens

    # Update the `special_tokens_map` that was loaded from the `special_tokens_map.json` file:
    special_tokens_map["additional_special_tokens"] = add_special_tokens

    # Update the `added_tokens` that was loaded from the `added_tokens.json` file:
    added_tokens = {
        t: tokenizer.convert_tokens_to_ids(t) for t in ["<mask>", "dyu_Latn", "fra_Latn"]
    }

    # Create a folder to save the new tokenizer:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    new_tokenizer_dir = Path(save_dir)

    # Save the new `tokenizer_config.json` file:
    with open(new_tokenizer_dir/"tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    # Save the `special_tokens_map.json` file:
    with open(new_tokenizer_dir/"special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f)

    # Save the `added_tokens.json` file:
    with open(new_tokenizer_dir/"added_tokens.json", "w") as f:
        json.dump(added_tokens, f)

    # Update the pre-trained NllbTokenizer's `sentencepiece` model:
    m = sp_model.ModelProto()
    m.ParseFromString(open("/tmp/sentencepiece.bpe.model", "rb").read())

    # Loop over `m.pieces` and keep only keep the tokens that are in the new vocab.
    # This takes a few minutes.
    seen = set()
    while True:
        if m.pieces[0].piece in seen:
            break
        x = m.pieces.pop(0)
        seen.add(x.piece)
        if x.piece in tokenizer.vocab:
            m.pieces.append(x)

    # Add tokens that were not in the old tokenizer's vocab:
    add_tokens = set(tokenizer.vocab.keys()) - seen
    for token in add_tokens:
        new_token = sp_model.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        m.pieces.append(new_token)
    assert len(m.pieces) == len(tokenizer.vocab)

    # Serialize the sentencepiece model
    with open(new_tokenizer_dir/'sentencepiece.bpe.model', 'wb') as f:
        f.write(m.SerializeToString())

    # Test loading the new tokenizer
    sp = spm.SentencePieceProcessor()
    assert sp.load(str(new_tokenizer_dir/'sentencepiece.bpe.model'))

    # Test loading the new tokenizer with `transformers`:
    try:
        tokenizer = NllbTokenizer.from_pretrained(new_tokenizer_dir)
    except Exception:
        print("Cannot load tokenizer with `transformers`")
