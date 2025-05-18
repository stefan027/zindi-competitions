# Lelapa AI Buzuzu-Mavi Challenge

This repository contains code and resources for the Lelapa AI Buzuzu-Mavi Challenge, aimed at creating smaller and smarter language models for African languages, specifically Swahili and Hausa. Below is a guide to setting up the environment, running the code, and understanding the workflow.

## Folder Structure and File Locations

- **`requirements_buzuzu_mavi.txt`**: Dependencies to be installed.
- **`create_instruct_data.ipynb`**: Generates instruction-tuning data for the model.
- **`modify_tokenizers.ipynb`**: Customizes tokenizers for Swahili and Hausa.
- **`prune.ipynb`**: Prunes the model to reduce its size while maintaining performance.
- **`finetune.ipynb`**: Fine-tunes the model on specific tasks.
- **`inference.ipynb`**: Runs inference to evaluate the model's performance.
- **`inkuba_instruct_sample.json`**: Instruction-tuning data (produced by `create_instruct_data.ipynb`)
- **`data`**
  - MTTrain.csv
  - MTTest.csv
  - SentimentTrain.csv
  - SentimentTest.csv
  - XNLITrain.csv
  - XNLITest.csv
- **`tokenizer`**: contains files for the modified tokenizer.
- **`weights`**:
  - vulavula_inkuba_instruct_tokenizer8k_pruned_v4.pth
  - vulavula_inkuba_instruct_tokenizer60k_pruned.pth

## Approach
InkubaLM has 420M parameters. With a vocabulary of 61,788 and embedding dimension of 2048, it means the first and last layers of the model alone account for 253M parameters, or 60% of the total number of parameters. In contrast, each hidden layer consists of only 16M parameters. Therefore, to make the model materially smaller, the number of parameters in the first and last layers must be reduced. My approach consisted of the following:
 1. Reduce the vocabulary size by only keeping frequently occuring tokens. The vocabulary size was reduced to 8,064 tokens. This change alone reduces the number of parameters by 220M parameters. InkubaLM's parameter weights are kept - we simply remove tokens that don't occur frequently and let the tokenizer fall back to smaller units.
 2. Reduce the dimension of all layers by a factor of two. For example, all 2048x2048 parameter tensors become 1024x1024. The pruning is achieved by randomly eliminating parameters.
 3. Alter the architecture so that the first embedding layer and the final prediction layer share the same parameters.
 4. (Optional): Reduce the number of hidden layers from 8 to 6. The code is written to always keep the first _N_ layers.

## Trained models

The following models were trained:
 - **`50M`**:
   - Number of parameters: **50,480,128**
   - Private LB score: 0.496534562
   - Feaures:
     - Modified tokenizer
     - Dimensions reduced by a factor of two
     - First and last layers share parameters
 - **`100M`**:
   - Number of parameters: **105,493,504**
   - Private LB score: 0.521065192
   - Feaures:
     - Original InkubaLM tokenizer
     - Dimensions reduced by a factor of two
     - First and last layers share parameters
 - **`40M`**:
   - Number of parameters: **42,087,424**
   - Private LB score: 0.532638201
   - Feaures:
     - Modified tokenizer
     - Dimensions reduced by a factor of two
     - First and last layers share parameters

All three models were submitted during the competition. The `50M` and `100M` models were my two selections for consideration. The `100M` was identified by Zindi as the best-performing, but that did not take the number of parameters into account.

I believe the `40M` can not be considered for the competition because it wasn't one of my two selected submissions. I will share the weights because it is both the smallest and most performant model. Additionally, a two-stage training process was utilised where it was first fine-tuned on translation data only, and then further fine-tuned on all three tasks. It makes the model particularly strong at translation while still performing well on the other tasks.


## Order of Execution

Run the notebooks in the following order:

1. **`create_instruct_data.ipynb`**: Prepares the data for instruction tuning.
  - Make sure you Huggingface token is set
  - Make sure `basedir` points to the correct directory
2. **`modify_tokenizers.ipynb`**: Adapts tokenizers for the target languages.
  - Make sure `basedir` points to the correct directory
3. **`prune.ipynb`**: Reduces the model size.
  - Make sure `basedir` points to the correct directory
  - Make sure the paths in `model_configs` are correct
  - Set `model_type` to 50M, 100M or 42M to prune to the required size
4. **`finetune_{model-version}.ipynb`**: Fine-tunes the pruned model.
  - Make sure `basedir`, `data_path`, and `tokenizer_path`
  - Make sure the weights to initialise the pruned model are in the base folder (i.e., same folder as the notebooks)
5. **`inference.ipynb`**: Evaluates the final model.
  - Make sure you Huggingface token is set if you did not run the training notebooks.
  - Make sure `output_path` and `data_path` are correct.
  

## Environment Setup

Use the provided `requirements_buzuzu_mavi.txt` file to set up the environment with `pip`:

```bash
pip install -r requirements_buzuzu_mavi.txt
```

## Hardware Requirements

The models were trained on Paperspace Gradient notebooks with a NVIDIA A6000 GPU.

## Credit
I learned a lot from the [source code](https://github.com/rasbt/LLMs-from-scratch/tree/main) that accompanies the book [Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167?crid=228R4JI0P0QFR&dib=eyJ2IjoiMSJ9.XvZyIer9iV133BWXqNiVt_OOJXZheO54dvZtQly8MC25PNYZrN3OWsGLjbg3I0G9hI3LkjwhsORxvHIob3nvCZFgdSSQEFe07VkehijGxT03n4Amdw7lnXxnsOUuWXeglfHnewCcV3DjL9zWHELfh5DG1ZErzFym3S6ZxSuFzNvoPkaq0uDlD_CKwqHdC0KM_RdvIqF0_2RudgvzRli0V155KkusHRck3pG7ybp5VyqKDC_GgL_MEywLwLhFgX6kOCgV6Rq90eTgSHFd6ac8krpIYjsHWe6H3IXbfKGvMXc.473O1-iUZC0z2hdx8L5Z5ZTNxtNV9gNPw_mE7QZ5Y90&dib_tag=se&keywords=raschka&qid=1730250834&sprefix=raschk,aps,162&sr=8-1&linkCode=sl1&tag=rasbt03-20&linkId=84ee23afbd12067e4098443718842dac&language=en_US&ref_=as_li_ss_tl). My training code is heavily influenced by [this notebook](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/01_main-chapter-code) from the repo.

> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
