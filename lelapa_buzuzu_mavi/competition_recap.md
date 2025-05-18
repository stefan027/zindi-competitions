## Introduction
Thank you to Zindi and Lelapa for hosting a competition on such an important topic. Congratulations to @yvan_carre whose private LB score is on a different level to the rest of us!

This competition challenged us to not only improve the model's accuracy but also make it smaller. I began by instruction-tuning InkubaLM to get a sense of what score might be achievable. After getting to about 0.45 on the public LB, I decided to start focussing on making InkubaLM smaller. Consider the scoring formula:

```
zindi_score = (PrivateLB_score + (1-(size/PARAM_SIZE))*PrivateLB_score )/2
```
My understanding of the formula is that a full InkubaLM with 0.4B parameters achieving an LB score of 0.5 will achieve `zindi_score = (0.5 + (1-1)*0.5)/2 = (0.5 + 0)/2 = 0.25`. In comparison, a model with half the number of parameters and a LB score of 0.4 will achieve `zindi_score = (0.4 + (1-0.5)*0.4)/2 = (0.4 + 0.2)/2 = 0.3`.

As a sidenote, it has been a little frustrating that the application of the scoring formula hasn't been more transparent, and that the private LB is still only based on F1 scores.

## Data
In addition to the data provided in the competition's Data page, I also used data from [Inkuba-Instruct](https://huggingface.co/datasets/lelapa/Inkuba-instruct) and [XNLI](https://huggingface.co/datasets/facebook/xnli). I'm still not sure if these datasets were considered 'external' - the data page refers to the competition data only as 'samples' - but I thought using it was allowed regardless given [this clarification](https://discord.com/channels/1252526072017326181/1283352485926277130/1332596707506258031) on the Lelapa Discord server.

## Approach

InkubaLM has 420M parameters. With a vocabulary of 61,788 and embedding dimension of 2048, it means that the first and last layers of the model alone account for 253M parameters, or 60% of the total number of parameters. In contrast, each hidden layer consists of only 16M parameters. Therefore, to make the model significantly smaller, the number of parameters in the first and last layers must be reduced. My approach consisted of the following:
 1. Reduce the vocabulary size by only keeping frequently occuring tokens. The vocabulary size was reduced to 8,064 tokens. This change alone reduced the number of parameters by 220M parameters. InkubaLM's parameter weights are kept - I simply remove tokens that don't occur frequently and let the tokenizer 'fall back' to smaller units.
 2. Reduce the dimension of all layers by a factor of two. For example, all 2048x2048 parameter tensors become 1024x1024. The pruning was achieved by randomly eliminating parameters. As with the embedding layer, I eliminated parameters but kept the weights of the remaining parameters.
 3. Alter the architecture so that the first embedding layer and the final prediction layer share the same parameters.
 4. (Optional): Reduce the number of hidden layers from 8 to 6. The code is written to always keep the first _N_ layers when pruning layers.
---

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
     - Reduce number of hidden layers from 8 to 6

All three models were submitted during the competition. The `50M` and `100M` models were my two selections for consideration. The `100M` was identified by the private LB as the best-performing, but that did not take the number of parameters into account. I believe that the `50M` model was the best submission when the number of parameters are considered.

The `40M` could not be considered for the competition because it wasn't one of my two selected submissions. I only trained it right at the end of the competition and didn't have time to test it thoroughly, and was therefore hesitant to select it as a submission. For me this is the most exciting model because it is both the smallest and most performant. Additionally, for the `40M` model, I used a two-stage training process where it was first fine-tuned on translation data only, and then further fine-tuned on all three tasks. It makes the model particularly strong at translation while still performing well on the other tasks.

## Code
All the code is on Github: .