"""Evaluation functions"""

import sacrebleu as scb
from fastai.learner import Metric
from utils import get_example, cleanup
from tqdm.auto import tqdm


def get_model_bleu_score(predictions=list[str], references=list[str]) -> float:
    """
    This function is copied from the competition's self-assessment tool.

    Args:
        predictions (list[str]): A list of the model's predicted translations.
        references (list[str]): A list of correct reference translations.

    Returns:
        float: The model's BLEU score (indicating translation performance).
    """
    sacrebleu = scb.corpus_bleu(
        hypotheses=predictions, references=[references]
    )
    return sacrebleu.score


def calculate_bleu(
    model, dataset, translate_func, src_lang="dyu_Latn", tgt_lang="fra_Latn",
    preproc_func=None, bs=128
):
    """
    Calculate the BLEU score over a `dataset`.

    Args:
        model (`transformers.PreTrainedModel`):
            The translation model to evaluate.
        dataset (`datasets.Dataset`):
            The dataset over which to calculate the BLEU score.
        translate_func (`Callable`):
            A function that translates a batch of examples using the model.
        src_lang (`str`):
            Source language code. Defaults to "dyu_Latn".
        tgt_lang (`str`):
            Target language code. Defaults to "fra_Latn".
        preproc_func (`Callable`, optional):
            A preprocessing function applied to the input examples and references.
            Defaults to `None`, which means no preprocessing is applied.
        bs (`int`):
            Batch size for processing the dataset. Defaults to 128.

    Returns:
        float:
            The BLEU score of the model over the provided dataset.
    """
    src = "fr" if src_lang == "fra_Latn" else "dyu"
    tgt = "fr" if tgt_lang == "fra_Latn" else "dyu"
    if preproc_func is None:
        def preproc_func(x):
            return x
    model.eval()
    cleanup()
    predictions, references = [], []
    xb, yb = [], []
    for i in tqdm(range(len(dataset))):
        example, reference = map(
            preproc_func, get_example(dataset, idx=i, return_reference=True, src=src, tgt=tgt)
        )
        xb.append(example)
        yb.append(reference)
        if len(xb) == bs or i == len(dataset)-1:
            predictions += translate_func(xb, src_lang=src_lang, tgt_lang=tgt_lang)
            references += yb
            xb, yb = [], []
    return get_model_bleu_score(predictions=predictions, references=references)


class CalculateBleu(Metric):
    """
    A fastai `Metric` class to calculate the validation BLEU score during training.

    This metric is used to track the model's translation performance across training 
    epochs by calculating the BLEU score on the validation set. It uses the provided 
    `translate_func` to generate predictions for validation samples and compares 
    them against reference translations to compute the BLEU score.

    Args:
        df (`pandas.DataFrame`):
            A DataFrame containing validation data. Each row represents a translation 
            pair with the first column being the source text and the second column 
            being the reference translation.
        translate_func (`Callable`):
            A function that translates a list of examples using the model.
        preproc_func (`Callable`):
            A preprocessing function applied to both source and reference texts 
            before translation and comparison.
    """
    def __init__(self, df, translate_func, preproc_func):
        self.func = get_model_bleu_score
        self.df = df
        self.translate_func = translate_func
        self.preproc_func = preproc_func

    def reset(self):
        self.predictions, self.references = [], []

    def accumulate(self, learn):
        idx = list(learn.yb[0]["indexes"].detach().cpu().numpy())
        b = self.df.iloc[idx]
        xb = b.iloc[:, 0].apply(self.preproc_func).tolist()
        yb = b.iloc[:, 1].apply(self.preproc_func).tolist()
        self.predictions += self.translate_func(xb)
        self.references += yb

    @property
    def value(self):
        if self.predictions:
            return self.func(predictions=self.predictions, references=self.references)
        return None

    @property
    def name(self):
        return "BLEU"
