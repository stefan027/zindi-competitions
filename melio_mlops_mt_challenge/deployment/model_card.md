## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

A lightweight machine translation model that translates Dyula to French.

- **Model type:** Neural Machine Translation
- **Language(s) (NLP):** Dyula to French
- **Finetuned from model [optional]:** NLLB-200

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
The model is intended to translate short sentences and phrases from Dyula to French.

The model is intended to be deployed where hardware is constrained. The final model uses less than 20MB RAM when loaded into memory.

### Example Payload

Here is an example payload for model inference.

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "i tɔgɔ bi cogodɔ"
            ]
        }
    ]
}
```

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Dyula is a low-resource language. A desktop search identified only two open-source pre-trained models that were trained on both Dyula and French. Our tests found `NLLB-200` to be the best-performing pre-trained model. The 600M parameter version of `NLLB-200` achieves a BLEU score of only 4.4 on our validation set. While our model performs improves the translation quality (see the `Evaluation` section below for details), translation quality might vary.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Translations should be interpreted as relaying the gist of the source text.

Training on back-translated data improved model accuracy. We believe training on additional Dyula-to-French data will improve model performance.

## How to Get Started with the Model

Refer to the `README` in the `deployment` folder for deployment instructions.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on a Dyula-French translation dataset that was created by [data354](https://data354.com/en/) and made available through a [Zindi competition](https://zindi.africa/competitions/melio-mlops-competition). The dataset is available on the [Hugging Face Hub](https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr). The following information is from the dataset card:

#### Overview

The Koumankan4Dyula corpus consists of 10929 pairs of Dioula-French sentences. This corpus is part of the Koumankan project, which proposes a scalable and cost-effective method for extending the CommonVoice dataset to the Dyula language and other African languages.

#### Data Splits

| Split | Proportion | Nr of examples |
|------ | ---------: | -------------: |
| Train |        73% |          8 065 |
| Valid |        14% |          1 471 |
| Test  |        13% |          1 393 |

### Training Procedure
<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
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

## Evaluation

The SacreBLEU score on the validation set for the various models are shown the table below. Some variation across training runs is expected.

| Model                                           | Validation BLEU |
| ----------------------------------------------- | --------------: |
| NLLB-200-distilled-600MB                        |             4.4 |
| NLLB-200-distilled-600MB fine-tuned             |            11.3 |
| Small model trained from scratch                |             7.3 |
| Small student-teacher distilled model (6 layer) |            10.3 |
| Small student-teacher distilled model (3 layer) |             9.1 |

## Technical Specifications

| Model | Vocab size | Embedding dimension | Hidden size | # layers | # parameters | Memory footprint (GB) |
|------------------------------------------------ | ------: | --: | --: | --: | --: | --: |
| NLLB-200-distilled-600MB                        | 256 206 | 1 024 | 4 096 | 12 | 615M | 2.30GB |
| NLLB-200-distilled-600MB fine-tuned             |  10 806 | 1 024 | 4 096 | 12 | 356M | 1.34GB |
| Small model trained from scratch                |   2 001 |   256 | 1 024 |  3 |   6M | 0.02GB |
| Small student-teacher distilled model (6 layer) |   3 710 |   512 | 2 048 |  6 |  46M | 0.17GB |
| Small student-teacher distilled model (3 layer) |   3 710 |   256 | 1 024 |  3 | --6M | 0.02GB |

### Compute Infrastructure

#### Training
All models can be trained on Google Colab's free T4 GPU or Kaggle's P100 GPU. A full end-to-end model run takes approximately 9h30 on a Kaggle P100 GPU. The total training time can be improved by approximately 30% by utilising distributed training on Kaggle's 2xT4 instances. A notebook showing how to leverage distributed training is provided in the code repository.

#### Inference
The small 3-layer trained models can run inference on a single CPU and less than 500MB RAM.
