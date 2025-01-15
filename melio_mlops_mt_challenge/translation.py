"""Translation functions"""

import torch
from utils import cleanup, preproc
from tqdm.auto import tqdm


def translate(
    article, model, tokenizer, src_lang="dyu_Latn", tgt_lang="fra_Latn",
    max_length=30, num_beams=1, do_sample=False, temperature=None
):
    """
    Translates a given text using a specified model and tokenizer.

    Args:
        article (`str`):
            The text to be translated from the source language to the target language.
        model (`transformers.PreTrainedModel`):
            The pre-trained model used for generating translations.
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer used for encoding the input text and decoding the generated output.
        src_lang (`str`, optional):
            The source language code for the tokenizer. Default is `"dyu_Latn"`.
        tgt_lang (`str`, optional):
            The target language code for the tokenizer. Default is `"fra_Latn"`.
        max_length (`int`, optional):
            The maximum length of the generated translation. Default is `30`.
        num_beams (`int`, optional):
            The number of beams for beam search. Default is `1`. If set to `1`, it uses
            greedy decoding.
        do_sample (`bool`, optional):
            Whether to use sampling instead of greedy decoding. Default is `False`.
        temperature (`float`, optional):
            The temperature to use for sampling. Higher values (e.g., 1.0) result in more
            diverse outputs, while lower values (e.g., 0.5) make the output more deterministic.
            Default is `None`.

    Returns:
        list of `str`: 
            The translated text(s) as a list of strings. If the input is a single string,
            the output will be a list with one translation.
    """
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        article, return_tensors='pt', padding=True, truncation=True, max_length=128
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_length=30,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def back_translate(
    ds, model, tokenizer, split="train", batch_size=16, src_lang="fra_Latn",
    tgt_lang="dyu_Latn", sample_src=True, src2src_model=None, src2src_tokenizer=None
):
    """
    Generate back-translations from a source language to a target language to augment the
    training data.

    This function first optionally generates source language to source language translations
    to introduce diversity into the training examples. It then generates translations from
    the source language to the target language.

    Args:
        ds (`Dataset`):
            A Huggingface dataset object containing the source language examples.
        model (`PreTrainedModel`):
            The model used to generate back-translations.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer used for encoding and decoding the translations.
        split (`str`, optional):
            The dataset split to use (e.g., 'train', 'validation'). Default is `"train"`.
        batch_size (`int`, optional):
            The number of examples to process in each batch. Default is `16`.
        src_lang (`str`, optional):
            The source language code (e.g., `'dyu_Latn'`, `'fra_Latn'`). Default is `"fra_Latn"`.
        tgt_lang (`str`, optional):
            The target language code (e.g., `'dyu_Latn'`, `'fra_Latn'`). Default is `"dyu_Latn"`.
        sample_src (`bool`, optional):
            Whether to first generate source language to source language translations before
            performing the back-translation. Default is `True`. Setting this to `True` can
            introduce more diversity into the training data.
        src2src_model (`PreTrainedModel`, optional):
            The model to use for generating source language to source language translations.
            If not provided, the main model is used.
        src2src_tokenizer (`PreTrainedTokenizer`, optional):
            The tokenizer to use for generating source language to source language translations.
            If not provided, the main tokenizer is used.

    Returns:
        tuple of list of `str`:
            - The back-translated examples in the target language.
            - The source language examples that were used to generate the back-translations.

    Notes:
        - This function assumes that the dataset `ds` has a `"translation"` field with source
          language examples.
        - The `preproc` function is applied to the text data before translation. Ensure that
          this function is defined and appropriate for the dataset.
        - The function uses multinomial sampling to generate more diverse translations. Adjust
          the `temperature`parameter in the `translate` function if needed to control the
          sampling behavior.
    """
    model.eval()
    if sample_src:
        src2src_model = model if src2src_model is None else src2src_model
        src2src_tokenizer = tokenizer if src2src_tokenizer is None else src2src_tokenizer
    cleanup()

    tgt, src = [], []
    xb = []
    ds_src_code = "fr" if src_lang == "fra_Latn" else "dyu"
    for i in tqdm(range(len(ds[split]))):
        example = ds[split][i]["translation"][ds_src_code]
        example = preproc(example)
        xb.append(example)
        if len(xb) == batch_size or i == len(ds[split])-1:
            if sample_src:
                source_sampled = translate(
                    xb, src2src_model, src2src_tokenizer, src_lang=src_lang, tgt_lang=src_lang,
                    do_sample=True, temperature=0.8
                )
                source_sampled = list(map(preproc, source_sampled))
            else:
                source_sampled = xb
            src += source_sampled
            tgt += translate(
                source_sampled, model, tokenizer, src_lang=src_lang, tgt_lang=tgt_lang,
                do_sample=True, temperature=0.2
            )
            xb = []
    return tgt, src
