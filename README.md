# LayoutLM-wikipedia-ja Model

This is a [LayoutLM](https://doi.org/10.1145/3394486.3403172) model pretrained on texts in the Japanese language.

## Model Details

### Model Description

- **Developed by:** JRIRD
- **Model type:** LayoutLM
- **Language(s) (NLP):** Japanese
- **License:** [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- **Finetuned from model:** [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2)

## Uses

The model is primarily aimed at being fine-tuned on a token classification task. You can use the raw model for masked language modeling, although it is not the primary use case. Refer to [https://github.com/nishiwakikazutaka/shinra2022-task2_jrird](https://github.com/nishiwakikazutaka/shinra2022-task2_jrird) for instructions on how to fine-tune the model. Note that the linked repository is written in Japanese.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
>>> from transformers import BertJapaneseTokenizer, LayoutLMModel
>>> import torch

>>> tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
>>> model = LayoutLMModel.from_pretrained(MODEL_PATH)

>>> tokens = tokenizer.tokenize("こんにちは")  # ['こん', '##にち', '##は']
>>> normalized_token_boxes = [[637, 773, 693, 782], [693, 773, 749, 782], [749, 773, 775, 782]]
>>> # add bounding boxes of cls + sep tokens
>>> bbox = [[0, 0, 0, 0]] + normalized_token_boxes + [[1000, 1000, 1000, 1000]]

>>> input_ids = [tokenizer.cls_token_id] \
                + tokenizer.convert_tokens_to_ids(tokens) \
                + [tokenizer.sep_token_id]
>>> attention_mask = [1] * len(input_ids)
>>> token_type_ids = [0] * len(input_ids)
>>> encoding = {
    "input_ids": torch.tensor([input_ids]),
    "attention_mask": torch.tensor([attention_mask]),
    "token_type_ids": torch.tensor([token_type_ids]),
    "bbox": torch.tensor([bbox]),
    }

>>> outputs = model(**encoding)
```

## Training Details

### Training Data

The model is trained on the Japanese version of Wikipedia. The training corpus is distributed as [training data of the SHINRA 2022 shared task](https://2022.shinra-project.info/data-download#subtask-common).

### Tokenization and Localization

We used the tokenizer of [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) to split texts into tokens (subwords). Each token is wrapped in a `<span>` tag with the no-wrap value set for the white-space property and localized by obtaining `BoundingClientRect`. The localization process was conducted with Google Chrome (106.0.5249.119) headless mode on Ubuntu 20.04.5 LTS with a 1,280*854 window size.

The vocabulary is the same as [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2).

### Training Procedure 

The model was trained using Masked Visual-Language Model (MVLM), but it was not trained using Multi-label Document Classification (MDC). We made this decision because we did not identify significant visual differences, such as those between a contract and an invoice, between the different Wikipedia articles.

#### Preprocessing

All parameters except the 2-D Position Embedding were initialized with weights from [cl-tohoku/bert-base-japanese-v2](https://huggingface.co/cl-tohoku/bert-base-japanese-v2). We initialized the 2-D Position Embedding with random values.

#### Training Hyperparameters

The model was trained on 8 NVIDIA A100 SXM4 GPUs for 100,000 steps, with a batch size of 256 with a maximum sequence length of 512. The optimizer used is Adam with a learning rate of 5e-5, &beta;<sub>1</sub>=0.9, &beta;<sub>2</sub>=0.999, learning rate warmup for 1,000 steps, and linear decay of the learning rate after. Additionally, we utilized fp16 mixed precision during training. The training took about 5.3 hours to finish.

## Evaluation

Our fine-tuned model achieved a macro-f1 score of 55.1451 on the leaderboard for the SHINRA 2022 shared task. You can check the leaderboard at [https://2022.shinra-project.info/#leaderboard](https://2022.shinra-project.info/#leaderboard) for detailed information.

## Citation

**BibTeX:**

```tex
@inproceedings{bibtex-id,
  title = {日本語情報抽出タスクのための{L}ayout{LM}モデルの評価},
  author = {西脇一尊 and 大沼俊輔 and 門脇一真},
  booktitle = {言語処理学会第29回年次大会(NLP2023)予稿集},
  year = {2023},
  pages = {522--527}
}
```
