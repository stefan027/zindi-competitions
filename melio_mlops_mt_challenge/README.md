## Getting started

All the code required to train the Dyula-to-French machine translation models submitted to the competition is contained in this repo.

> **Note: This submission used the 'small NLLB-200 model trained from scratch'.**

The recommended way to run the full training process is to run the `train_end_to_end.ipynb` notebook from top to bottom. The notebook can be run in Google Colab or Kaggle. Since free Colab sessions are prone to disconnect after a few hours, Kaggle is recommended to run everything end-to-end. Further instructions relevant to the particular platform is provided in the notebook.

The final trained model used for the submission will be saved in `./saved_models/nllb-dyu-fr-10MB`.

Individual notebooks with more descriptive information can be found in the `notebooks` folder. Please refer to the `Training procedure` section below for more details. These notebooks are not required to reproduce the competition results.

For detailed documention of the source code, please refer to the `docs` folder.

## Training Details

### Setup

```bash
pip install -r requirements.txt
```

### Rationale & Objective

The competition's evaluation criteria placed a heavy emphasis on low resource use (latency, memory usage, image size) during inference. While the BLEU score is weighted heavily in the scoring criteria, the mean BLEU score used in the normalisation formula de-emphasises its importance. Based on an analysis of the self-assessment tool that was provided, the mean BLEU score in the normalisation formula was chosen to be 40. This score seems high given the BLEU scores achieved by large models for better-resourced languages in the [NLLB project](https://arxiv.org/abs/2207.04672). The result is that the competition metric doesn't differentiate well, for example, between a BLEU score of 6.0 and a BLEU score of 10.0, and thus puts a premium on achieving high scores for the resource-use metrics.

The chosen training objective was therefore to distil as much performance as possible from the `NLLB-200-distilled-600M` model into a model that is as small as possible in order to score highly on the latency, memory usage, and image size metrics while still achieving a reasonable BLEU score.

Without fine-tuning, the `NLLB-200-distilled-600M` model achieved a BLEU score of only 4.4 on the competition's validation set. With fine-tuning, the validation BLEU score can be pushed up to approximately 12. However, the 600M parameter model has a memory footprint of 2.3GB.

Our training procedure managed to retain more than 80% of the translation quality while reducing the model to 6M parameters and a memory footprint of only 25MB in `float32` and 12.4MB in `bfloat16`.

Slightly larger models that are still able to run on the hardware limitations imposed by the competition can retain even more of the translation quality of the 600M parameter model, but suffers too much on the competition's resource-use metrics to be competitive.

### Training Data

The model was trained on a Dyula-French translation dataset that was created by [data354](https://data354.com/en/) and made available through a [Zindi competition](https://zindi.africa/competitions/melio-mlops-competition). The dataset is available on the [Hugging Face Hub](https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr). The following information is from the dataset card:

#### Overview

The Koumankan4Dyula corpus consists of 10,929 pairs of Dioula-French sentences. This corpus is part of the Koumankan project, which proposes a scalable and cost-effective method for extending the CommonVoice dataset to the Dyula language and other African languages.

#### Data Splits

| Split | Proportion | Nr of examples |
|------ | ---------: | -------------: |
| Train |        73% |          8 065 |
| Valid |        14% |          1 471 |
| Test  |        13% |          1 393 |

### Training Procedure

Given the small amount of training data that was available, the training procedure was designed to extract as much information from the pre-trained `NLLB-200` model as possible. Given the hardware constraints on model deployment, a simple fine-tune of the model was not feasible.

The training procedure consisted of five stages:
 1. Modify the `NllbTokenizer`s: By reducing the vocabulary of the tokenizers, the size of the embedding matrix is reduced considerably. We create three tokenizers for different purposes:
    - A version of `NllbTokenizer` with the vocabulary limited to tokens that are present in our training data.
    - A version of `NllbTokenizer` with the vocabulary limited to tokens that appear at least 5 times in our training data.
    - A `NllbTokenizer` with a vocabulary of only 2000 tokens trained on our training data.
 2. Create two fine-tunes of `Nllb-200-distilled-600M` that are then used to generate back-translations to expand the training dataset. Using back-translations from larger models allows us to 'distil' some knowledge from the larger models to the small downstream models.
 3. Use the fine-tuned models from step 2 to generate synthetic training data.
 4. Train a small (< 20MB memory footprint) model from scratch on the original training data and the back-translations from step 3.
 5. Train another small (< 20MB memory footprint) model using a teacher-student distillation training process on the original training data and the back-translations from step 3.

The stage 5 training step above consists of the following steps:
 1. Fine-tune NLLB-200-600M, but with a modified tokenizer with a smaller vocabulary
 2. Use the model from step 1 as teacher. The student is a pruned version of the model where half of the encoder and decoder layers are dropped, and half of the neurons in the embedding and linear layers are dropped.
 3. Repeat step 2 so that the student model is again down-scaled by a factor of 2.

**Note: This submission used the 'small NLLB-200 model trained from scratch' (Step 4 in the trainin procedure outlined above).** The distillation model (Step 5 in the training procedure outlined above) is therefore not trained in this configuration of the `train_end_to_end.ipynb` notebook. This is achieved by setting `TRAIN_DYU_FRA_DISTILLED = False` in the notebook.

The individual notebooks that perform each of the stages described above are listed in the table below:

| Training step                         | Notebooks                       |
|-------------------------------------- | ------------------------------- |
| Modify tokenizers                     | `limit_tokenizer_vocab.ipynb`   |
|                                       | `train_tokenizer.ipynb`         |
| Large DYU-FRA fine-tune               | `fine_tune_dyu_fra.ipynb`       |
| Large FRA-DYU fine-tune               | `fine_tune_fra_dyu.ipynb`       |
| Generate back-translated data         | `create_backtranslations.ipynb` |
| Train small model from scratch        | `train.ipynb`                   |
| Teacher-student distillation training | `train-distil.ipynb`            |

#### Preprocessing

Some special characters are removed and text is converted to lowercase for model training and inference.

```python
CHARS_TO_REMOVE_REGEX = r'[!"&\(\),-./:;=?+.\n\[\]]'
```

#### Training Hyperparameters

##### Larger Dyula-to-French model used to generate back-translations
Trained for 10 epochs using [1cycle training](https://arxiv.org/abs/1708.07120) with an initial learning rate of 1e-4.

##### Larger French-to-Dyula model used to generate back-translations
Trained for 3 epochs using [1cycle training](https://arxiv.org/abs/1708.07120) with an initial learning rate of 1e-4.

##### Small (< 20MB memory footprint) model from scratch
The model is first trained on the original and synthetic training data using the [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) schedule with `n_cycles=5`, `cycle_len=1`, `lr_max=1e-3` and `cycle_mult=2` for a total of 31 epochs.

The model is then fine-tuned on the original training data for 10 epochs with an initial learning rate of 1e-4, and cosine annealing after 7.5 epochs.

Finally, the model is converted to `bfloat16` for inference.

##### Small (< 20MB memory footprint) distilled model
 - Step 1: Fine-tune NLLB-200-distilled-600m with its vocabulary reduced to 3,710 tokens on the original training data only. Training is done for 10 epochs with an initial learning rate of 1e-4, and cosine annealing starting after 7.5 epochs.
 - Step 2: Teacher-student distillation training on the original and synthetic training data using the [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) schedule with `n_cycles=5`, `cycle_len=1`, `lr_max=1e-3` and `cycle_mult=2` for a total of 31 epochs.
 - Step 3: Teacher-student distillation training as per step 2. The model is then fine-tuned on the original training data for 10 epochs with an initial learning rate of 1e-5, and cosine annealing after 7.5 epochs.

## Saved model
The trained model used for the submission will be saved as `./saved_models/nllb-dyu-fr-10MB`.

## A note on reproducibility
Random seeds were fixed for `torch`, `numpy` and `random`. Exact library versions are specified in the `requirements.txt` file. Despite this, some randomness remain. Based on my experiments, I don't believe this residual randomness results in any material impact on inference-time performance in terms of the BLEU score.

## Evaluation

The SacreBLEU score on the validation set for the various models are shown the table below. Some variation across training runs is expected.

| Model                                           | Validation BLEU |
| ----------------------------------------------- | --------------: |
| NLLB-200-distilled-600MB                        |             4.4 |
| NLLB-200-distilled-600MB fine-tuned             |            11.3 |
| Small model trained from scratch                |             7.3 |
| Small student-teacher distilled model (6 layer) |            10.3 |
| Small student-teacher distilled model (3 layer) |             9.1 |

### Zindi leaderboard scores

| Model                                           | Zindi submission ID | Public LB | Private LB |
| ----------------------------------------------- | ------------------- | --------: | ---------: |
| Small model trained from scratch                |            4cQTFULr |    0.4681 |     0.8175 |
| Small student-teacher distilled model (3 layer) |            CNEyC59W |    0.4675 |    Unknown |

## Technical Specifications

| Model | Vocab size | Embedding dimension | Hidden size | # layers | # parameters | Memory footprint (GB) |
|------------------------------------------------ | ------: | --: | --: | --: | --: | --: |
| NLLB-200-distilled-600MB                        | 256 206 | 1 024 | 4 096 | 12 | 615M | 2.30GB |
| NLLB-200-distilled-600MB fine-tuned             |  10 806 | 1 024 | 4 096 | 12 | 356M | 1.34GB |
| Small model trained from scratch                |   2 001 |   256 | 1 024 |  3 |   6M | 0.02GB |
| Small student-teacher distilled model (6 layer) |   3 710 |   512 | 2 048 |  6 |  46M | 0.17GB |
| Small student-teacher distilled model (3 layer) |   3 710 |   256 | 1 024 |  3 |   6M | 0.02GB |

### Compute Infrastructure

All models can be trained on Google Colab's free T4 GPU or Kaggle's P100 GPU.  The total training time can be improved by approximately 30% by utilising distributed training on Kaggle's 2xT4 instances. A notebook showing how to leverage distributed training is provided in the code repository.

A full end-to-end model run for the submission model takes approximately 4h on a Kaggle P100 GPU instance. The time for each step is given in the table below:

| Training step                         | Time on a Kaggle P100 GPU instance (hh:mm:ss) |
|-------------------------------------- | -----------------------------------------: |
| Train tokenizer 1                     |                                  00:10:40  |
| Train tokenizer 2                     |                                       N/A  |
| Train tokenizer 3                     |                                  00:13:00  |
| Train large DYU-FRA fine-tune         |                                  00:22:50  |
| Train large FRA-DYU fine-tune         |                                  00:07:10  |
| Generate back-translated data         |                                  02:05:00  |
| Train small model from scratch        |                                  00:40:20  |
| Teacher-student distillation training |                                       N/A  |
| **Total wall time**                   |                              **03:49:00**  |

## Deployment & Inference
See the `README` in the `deployment` folder.