"""Functions to load, initialize models and trainers"""

from copy import deepcopy
from functools import partial
from collections import OrderedDict
import random

import torch
from torch import nn
from torch import float32
import torch.nn.functional as F
import transformers
from transformers import (
    M2M100ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    NllbTokenizer
)
from fastai.learner import Learner
from fastai.callback.schedule import lr_find

from translation import translate
from evaluation import CalculateBleu


def remap_model_embeddings(model, tokenizer, tokenizer_old, init_embeds_for_new_tokens=False):
    """
    Remaps the embedding weights of a `model` that was trained with `tokenizer_old`
    to match the positions of the tokens in a new vocabulary provided by `tokenizer`.

    Args:
        model (`transformers.M2M100ForConditionalGeneration`):
            The HuggingFace model to have its embeddings remapped.
        tokenizer (`transformers.NllbTokenizer`):
            The tokenizer that provides the new vocabulary for the model.
        tokenizer_old (`transformers.NllbTokenizer`):
            The tokenizer that was originally used to train the model.
        init_embeds_for_new_tokens (`bool`):
            If True, generates random embeddings for tokens present in the
            new tokenizer but absent in the old tokenizer. If False, these
            new tokens will inherit the embedding of the <unk> token.

    Returns:
        `transformers.M2M100ForConditionalGeneration`:
            The model with remapped embeddings, loaded on CUDA, with potentially
            new random embeddings for new tokens.
    """
    # Get all the tokens in the new vocabulary
    tokens = tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size)))
    # Find the indexes of the tokens in the old vocabulary
    embed_keep_rows = tokenizer_old.convert_tokens_to_ids(tokens)

    # If more than one new token is mapped to the <unk> token, it means that we
    # have some new tokens in the new tokenizer. We'll initialise random embeddings
    # for those tokens if `init_embeds_for_new_tokens == True`. If not, the initial
    # embeddings for the new tokens default to the embedding for the <unk> token.
    new_token_rows = []
    unk_id_old = tokenizer_old.unk_token_id
    unk_id_new = tokenizer.unk_token_id
    # Get the indexes in the new vocabulary of where we must replace embeddings with
    # new random embeddings
    new_token_rows = [
        i for i, r in enumerate(embed_keep_rows) if r == unk_id_old and i != unk_id_new
    ]
    print("Number of tokens w/o embeddings:", len(new_token_rows))
    # Generate random embeddings for new tokens
    if init_embeds_for_new_tokens and new_token_rows:
        new_token_embeds = nn.Linear(
            model.config.d_model, len(new_token_rows), device=model.device).weight
        nn.init.xavier_normal_(new_token_embeds)

    weights = model.state_dict()

    # Find all the embedding weights in the model state_dict
    embed_params = []
    for k, v in weights.items():
        if v.shape[0] == model.config.vocab_size:
            print(k, v.shape)
            embed_params.append(k)

    # Keep only embedding weights for tokens that are in the new vocabulary
    for p in embed_params:
        weights[p] = weights[p][embed_keep_rows].clone()
        if init_embeds_for_new_tokens and new_token_rows:
            weights[p][new_token_rows] = new_token_embeds.clone()

    # Generate the new model config from the existing model config
    new_config = deepcopy(model.config)
    new_config.vocab_size = tokenizer.vocab_size
    new_config.transformers_version = transformers.__version__

    # Create the new model from the new model config and load the weights
    model = M2M100ForConditionalGeneration(new_config).cuda()
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3 :.2f}GB")
    model.load_state_dict(weights)
    return model


def load_model(
    model_id, load_tokenizer=True, tokenizer_id=None, remap_embeddings=False,
    init_embeds_for_new_tokens=False, old_tokenizer_id=None, torch_dtype=float32,
    src_language=None, tgt_language=None
):
    """
    Loads a pretrained model and, optionally, a tokenizer. Optionally remaps model
    embeddings if a different tokenizer is used.

    Args:
        model_id (`str`):
            Path or ID of the pretrained model to load.
        load_tokenizer (`bool`):
            Whether to load and return a pretrained tokenizer. Default is True.
        tokenizer_id (`str`, optional):
            Path or ID of the tokenizer to load. Defaults to `model_id` if not specified.
        remap_embeddings (`bool`):
            Whether to remap embeddings if a different tokenizer is used than the one the
            model was originally trained with. Default is False.
        init_embeds_for_new_tokens (`bool`):
            If True, generates random embeddings for tokens present in the
            new tokenizer but absent in the old tokenizer. If False, these
            new tokens will inherit the embedding of the <unk> token.
            Default is False.
        old_tokenizer_id (`str`, optional):
            Path or ID of the tokenizer that was originally used to train the model.
            Only relevant if `remap_embeddings` is True.
        torch_dtype (`torch.dtype`):
            The data type to load the model with. Default is `torch.float32`.
        src_language (`str`, optional):
            The source language code for the tokenizer. Either 'dyu_Latn' or 'fra_Latn'.
        tgt_language (`str`, optional):
            The target language code for the tokenizer. Either 'dyu_Latn' or 'fra_Latn'.

    Returns:
        `transformers.AutoModelForSeq2SeqLM`:
            The pretrained model, optionally with remapped embeddings.
        `transformers.NllbTokenizer`, optional:
            The pretrained tokenizer, if `load_tokenizer` is True.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch_dtype
    )

    if remap_embeddings and tokenizer_id is None:
        print("`remap_embeddings is True` but `tokenizer_id is None`. Cannot remap embeddings.")

    if remap_embeddings:
        old_tokenizer_id = model_id if old_tokenizer_id is None else old_tokenizer_id
        tokenizer = NllbTokenizer.from_pretrained(tokenizer_id)
        tokenizer_old = NllbTokenizer.from_pretrained(old_tokenizer_id)
        model = remap_model_embeddings(
            model, tokenizer, tokenizer_old, init_embeds_for_new_tokens=init_embeds_for_new_tokens
        )
        if load_tokenizer:
            return model, tokenizer
    elif load_tokenizer:
        tokenizer_id = model_id if tokenizer_id is None else tokenizer_id
        tokenizer = NllbTokenizer.from_pretrained(
            tokenizer_id, src_language=src_language, tgt_language=tgt_language
        )
        return model, tokenizer
    return model


def _update_config(old_value, div=None, new_value=None):
    "Update model config"
    if new_value is not None:
        return new_value
    assert old_value % div == 0
    return old_value // div


def instantiate_small_model(
    base_model_id, dim_factor=4, layer_factor=4, vocab_size=None,
    encoder_ffn_dim=None, encoder_layers=None, decoder_ffn_dim=None, decoder_layers=None,
    num_hidden_layers=None, d_model=None, max_position_embeddings=None
):
    """
    Instantiates a smaller version the base `M2M100ForConditionalGeneration` model. 
    The function adjusts various configuration parameters such as feed-forward
    dimensions, number of layers, hidden layers, and more.

    Args:
        base_model_id (`str`):
            The identifier of the base model to load from pre-trained models.
        dim_factor (`int`):
            The factor by which to scale the feed-forward dimensions of the encoder
            and decoder. Default is 4.
        layer_factor (`int`):
            The factor by which to scale the number of layers in the encoder and
            decoder. Default is 4.
        vocab_size (`int`, optional):
            The size of the vocabulary. If not provided, the original vocabulary
            size is used.
        encoder_ffn_dim (`int`, optional):
            Custom feed-forward dimension for the encoder. If not provided, the default
            value is scaled by `dim_factor`.
        encoder_layers (`int`, optional):
            Custom number of layers for the encoder. If not provided, the default number
            is scaled by `layer_factor`.
        decoder_ffn_dim (`int`, optional):
            Custom feed-forward dimension for the decoder. If not provided, the default
            value is scaled by `dim_factor`.
        decoder_layers (`int`, optional):
            Custom number of layers for the decoder. If not provided, the default number
            is scaled by `layer_factor`.
        num_hidden_layers (`int`, optional):
            Custom number of hidden layers. If not provided, the default number is scaled
            by `layer_factor`.
        d_model (`int`, optional):
            Custom size of the model's hidden states. If not provided, the default size
            is scaled by `dim_factor`.
        max_position_embeddings (`int`, optional):
            Custom maximum position embeddings. If not provided, the default number is
            scaled by `dim_factor`.

    Returns:
        `M2M100ForConditionalGeneration`:
            The modified model with updated configuration, loaded on CUDA.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, device_map="cuda")
    new_config = deepcopy(model.config)
    new_config.encoder_ffn_dim = _update_config(
        new_config.encoder_ffn_dim, dim_factor, encoder_ffn_dim
    )
    new_config.encoder_layers = _update_config(
        new_config.encoder_layers, layer_factor, encoder_layers
    )
    new_config.decoder_ffn_dim = _update_config(
        new_config.decoder_ffn_dim, dim_factor, decoder_ffn_dim
    )
    new_config.decoder_layers = _update_config(
        new_config.decoder_layers, layer_factor, decoder_layers
    )
    new_config.num_hidden_layers = _update_config(
        new_config.num_hidden_layers, layer_factor, num_hidden_layers
    )
    new_config.d_model = _update_config(
        new_config.d_model, dim_factor, d_model
    )
    new_config.max_position_embeddings = _update_config(
        new_config.max_position_embeddings, dim_factor, max_position_embeddings
    )
    if vocab_size is not None:
        new_config.vocab_size = vocab_size
    model = M2M100ForConditionalGeneration(new_config).cuda()
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3 :.2f}GB")
    return model


def _reduce_dim(model, size_factor=4, target_dims=[1024, 2048, 4096], strategy="average"):
    """
    Reduce dimensions that match those in `target_dims` by a factor of `size_factor`
    
    Args:
        model (`transformers.M2M100ForConditionalGeneration`):
            A `transformers.M2M100ForConditionalGeneration` model
        size_factor (`int`):
            The factor by which to scale the feed-forward dimensions
            of the encoder and decoder. Default is 4.
        target_dims (`list` of `int`):
            Reduces all dimensions that matches any of the sizes in the list.
        strategy (`str`):
            One of 'average', 'alternate' or 'random'
    """
    assert all(t%size_factor == 0 for t in target_dims),\
        "`target_dims` must be divisible by `size_factor`"
    weights = model.state_dict()
    for name, module in model.named_modules():
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            # print(name, module.weight.shape)
            w = weights[f"{name}.weight"]
            s = w.shape
            if s[0] in target_dims:
                if strategy == "average":
                    w_ = []
                    for i in range(size_factor):
                        mask = list(range(i, s[0], size_factor))
                        w_.append(w[mask])
                    w = sum(w_) / size_factor
                elif strategy == "alternate":
                    mask = list(range(0, s[0], size_factor))
                    w = w[mask]
                elif strategy == "random":
                    mask = random.sample(range(s[0]), s[0]//size_factor)
                    w = w[mask]
                else:
                    raise ValueError
            if len(s) == 2 and s[1] in target_dims:
                if strategy == "average":
                    w_ = []
                    for i in range(size_factor):
                        mask = list(range(i, s[1], size_factor))
                        w_.append(w[:, mask])
                    w = sum(w_) / size_factor
                elif strategy == "alternate":
                    mask = list(range(0, s[1], size_factor))
                    w = w[:, mask].clone()
                elif strategy == "random":
                    mask = random.sample(range(s[1]), s[1]//size_factor)
                    w = w[:, mask]
                else:
                    raise ValueError
            weights[f"{name}.weight"] = w.clone()

        if hasattr(module, "bias") and getattr(module, "bias") is not None:
            # print(name, module.bias.shape)
            b = weights[f"{name}.bias"]
            s = b.shape
            assert len(s) == 1
            if s[0] in target_dims:
                if strategy == "average":
                    b_ = []
                    for i in range(size_factor):
                        mask = list(range(i, s[0], size_factor))
                        b_.append(b[mask])
                    b = sum(b_) / size_factor
                elif strategy == "alternate":
                    mask = list(range(0, s[0], size_factor))
                    b = b[mask]
                elif strategy == "random":
                    mask = random.sample(range(s[0]), s[0]//size_factor)
                    b = b[mask]
                else:
                    raise ValueError
            weights[f"{name}.bias"] = b.clone()
    return weights


def _get_layer_number(name):
    "Get layer index from layer name"
    return int(name.split('.')[3])


def _new_layer_name(name, i):
    "Rename layer `name` by replacing the layer number with the value of `i`"
    pre = ".".join(name.split('.')[:3])
    post = ".".join(name.split('.')[4:])
    return pre + "." + str(i) + "." + post


def _reduce_num_layers(model, weights=None, layer_size_factor=3, strategy="alternate"):
    """
    Reduce the number of layers by a factor of `layer_size_factor`
    
    Args:
        model (`transformers.M2M100ForConditionalGeneration`):
            A `transformers.M2M100ForConditionalGeneration` model
        weights (`state_dict`):
            Optional `state_dict` for `model` that is the result of reducing
            dimensions using `_reduce_dim`
        layer_size_factor (`int`):
            The factor by which to scale the number of layers in the
            encoder and decoder. Default is 3.
        strategy (`str`):
            One of 'average' or 'alternate'
    """
    if weights is None:
        weights = model.state_dict()
    new_weights = OrderedDict()
    for name, module in model.named_modules():
        if hasattr(module, "weight") or hasattr(module, "bias"):
            if '.layers.' in name:
                i = _get_layer_number(name)
                new_i = i // layer_size_factor
                new_name = _new_layer_name(name, new_i)
                for k in ("weight", "bias"):
                    if hasattr(module, k) and getattr(module, k) is not None:
                        if strategy == "alternate":
                            if i % layer_size_factor == 0:
                                new_weights[f"{new_name}.{k}"] = weights[f"{name}.{k}"].clone()
                        elif strategy == "average":
                            if i % layer_size_factor == 0:
                                new_weights[f"{new_name}.{k}"] = weights[f"{name}.{k}"].clone()
                            else:
                                new_weights[f"{new_name}.{k}"] += weights[f"{name}.{k}"].clone()
                            if (i+1) % layer_size_factor == 0:
                                new_weights[f"{new_name}.{k}"] /= layer_size_factor
                        else:
                            raise ValueError
                # print(name, i, new_name)
            else:
                for k in ("weight", "bias"):
                    if hasattr(module, k) and getattr(module, k) is not None:
                        new_weights[f"{name}.{k}"] = weights[f"{name}.{k}"].clone()
    return new_weights


def prune(
    model, reduce_dims=True, reduce_layers=True, size_factor=4,
    target_dims=[1024, 2048, 4096], dim_strategy="average",
    layer_size_factor=3, layer_strategy="alternate"
):
    """
    Prunes the given `model` by reducing its dimensions and/or the number of layers,
    based on the specified factors.

    Args:
        model (`M2M100ForConditionalGeneration`):
            The pre-trained model to prune.
        reduce_dims (`bool`):
            Whether to reduce the model's dimensions (e.g., feed-forward dimensions
            and hidden states). Default is True.
        reduce_layers (`bool`):
            Whether to reduce the number of layers in the model. Default is True.
        size_factor (`int`):
            The factor by which to reduce dimensions in the model. Default is 4.
        target_dims (`list[int]`):
            A list of dimensions in the model to be reduced by the `size_factor`.
            Default is [1024, 2048, 4096].
        dim_strategy (`str`):
            The strategy for reducing dimensions ("average", "alternatee", "random").
            Default is "average".
        layer_size_factor (`int`):
            The factor by which to reduce the number of layers in the model. Default is 3.
        layer_strategy (`str`):
            The strategy for reducing layers ("average", "alternate"). Default is "alternate".

    Returns:
        `M2M100ForConditionalGeneration`:
            The pruned model with updated configuration, loaded on CUDA.
    """
    new_config = deepcopy(model.config)

    if reduce_dims:
        w = _reduce_dim(
            model, size_factor=size_factor, target_dims=target_dims,
            strategy=dim_strategy
        )
        new_config.encoder_ffn_dim //= size_factor
        new_config.decoder_ffn_dim //= size_factor
        new_config.d_model //= size_factor
        new_config.max_position_embeddings //= size_factor

    if reduce_layers:
        w = _reduce_num_layers(
            model, weights=w, layer_size_factor=layer_size_factor,
            strategy=layer_strategy
        )
        new_config.encoder_layers //= layer_size_factor
        new_config.decoder_layers //= layer_size_factor
        new_config.num_hidden_layers //= layer_size_factor

    model = M2M100ForConditionalGeneration(new_config).cuda()
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3 :.2f}GB")
    return model


class AdaptedModel(nn.Module):
    """
    A basic wrapper around a HuggingFace model that modifies the output to only\
    return the logits.

    Args:
        model (`PreTrainedModel`):
            A pre-trained HuggingFace model to be wrapped.
    """
    def __init__(self, model):
        super().__init__()
        self.hf_model = model

    def forward(self, inputs):
        return self.hf_model(**inputs).logits


def create_learner(dls, model, tokenizer, src_lang="dyu_Latn", tgt_lang="fra_Latn", wd=None):
    """
    Convenience function to create a fastai `Learner` from a HuggingFace `model`
    for translation tasks.

    Args:
        dls (`DataLoaders`):
            The DataLoaders object containing the training and validation datasets.
        model (`PreTrainedModel`):
            A pre-trained HuggingFace model used for translation.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer corresponding to the HuggingFace model.
        src_lang (`str`):
            The source language code for translation. Default is "dyu_Latn".
        tgt_lang (`str`):
            The target language code for translation. Default is "fra_Latn".
        wd (`float`, optional):
            The weight decay value used during training. Default is None.

    Returns:
        `Learner`:
            A fastai `Learner` object configured for training the HuggingFace model
            with cross-entropy loss and BLEU score as a metric.
    """
    translate_func = partial(
        translate, model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang
    )
    # Callback to calculate the validation BLEU score after each epoch
    bleu_score = CalculateBleu(
        dls.valid.dataset.df, translate_func, dls.valid.dataset.preproc_func
    )

    def _loss_func(lm_logits, labels):
        labels = labels["input_ids"]
        return nn.CrossEntropyLoss()(lm_logits.view(-1, model.config.vocab_size), labels.view(-1))

    pt_model = AdaptedModel(model)
    return Learner(dls, pt_model, loss_func=_loss_func, metrics=[bleu_score], wd=wd)


class DistilModel(nn.Module):
    """
    A wrapper around HuggingFace models used for distillation training. 
    Given a `student_model` and a `teacher_model`, it returns a tuple
    containing the logits from both models.

    Args:
        student_model (`PreTrainedModel`):
            The student model used for distillation training.
        teacher_model (`PreTrainedModel`):
            The teacher model used for distillation training, whose logits
            are returned without gradient updates.
    """
    def __init__(self, student_model, teacher_model):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

    def forward(self, inputs):
        student_logits = self.student_model(**inputs).logits
        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits
        return tuple((student_logits, teacher_logits))


class DistilLoss:
    """
    Calculates the distillation loss
    """
    def __init__(self, vocab_size, temperature=2., alpha=0.5):
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.alpha = alpha
        self.distil_loss_func = nn.KLDivLoss(reduction="batchmean")

    def lm_loss(self, lm_logits, labels):
        labels = labels["input_ids"]
        return nn.CrossEntropyLoss()(lm_logits.view(-1, self.vocab_size), labels.view(-1))

    def __call__(self, logits, labels):
        student_logits, teacher_logits = logits
        assert student_logits.shape == teacher_logits.shape
        student_loss = self.lm_loss(student_logits, labels)
        distil_loss = self.distil_loss_func(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1) * (self.temperature ** 2)
        )
        return self.alpha * student_loss + (1. - self.alpha) * distil_loss


def create_distillation_learner(
    dls, student_model, teacher_model, tokenizer,
    src_lang="dyu_Latn", tgt_lang="fra_Latn", wd=None
):
    """
    Convenience function to create a fastai `Learner` for distillation learning using a 
    `teacher_model` and `student_model`.

    Args:
        dls (`DataLoaders`):
            The DataLoaders object containing the training and validation datasets.
        student_model (`PreTrainedModel`):
            The student model that will be trained in the distillation process.
        teacher_model (`PreTrainedModel`):
            The teacher model providing the target logits for distillation.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer corresponding to the HuggingFace models.
        src_lang (`str`):
            The source language code for translation. Default is "dyu_Latn".
        tgt_lang (`str`):
            The target language code for translation. Default is "fra_Latn".
        wd (`float`, optional):
            The weight decay value used during training. Default is None.

    Returns:
        `Learner`:
            A fastai `Learner` object configured for distillation learning with
            cross-entropy loss, distillation loss, and BLEU score as a metric.
    """
    translate_func = partial(
        translate, model=student_model, tokenizer=tokenizer,
        src_lang="dyu_Latn", tgt_lang="fra_Latn"
    )
    # Callback to calculate the validation BLEU score after each epoch
    bleu_score = CalculateBleu(
        dls.valid.dataset.df, translate_func, dls.valid.dataset.preproc_func
    )

    pt_model = DistilModel(student_model, teacher_model)
    _loss_func = DistilLoss(tokenizer.vocab_size, temperature=2., alpha=0.5)
    return Learner(dls, pt_model, loss_func=_loss_func, metrics=[bleu_score], wd=None)
