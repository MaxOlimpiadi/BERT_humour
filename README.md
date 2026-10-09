# Humor Detection with Transformer Models

A research project fine-tuning pretrained transformer models for
**binary humor classification**.

The model is configurable through `MODEL_NAME`.

**Models and Training**

- **Labels:** `0` = non-humorous, `1` = humorous.
- **Optimizer:** AdamW with a learning rate of `5e-5`.
- **Batch size:** 16.
- **Maximum input length:** 512 tokens.
- **Maximum epochs:** 10.
- **Early stopping:** three epochs without validation-loss improvement.

All model parameters are fine-tuned. The best checkpoint is selected by validation loss and reloaded for test evaluation.

**Data**

Two datasets are used:

- **ColBERT dataset:** obtained from the
  [ColBERT repository](https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection),
  associated with the paper *ColBERT: Using BERT Sentence Embedding in Parallel
  Neural Networks for Computational Humor*.
- **Kaggle dataset:** obtained from Kaggle.

The preparation code creates stratified **80% training / 10% validation /
10% test** splits with `random_state=42`. 

Datasets and prepared splits are included in the repository.

**Project Structure**

```text
learning_model.py          # Main training and evaluation script
requirements_minimal.txt   # Dependencies
experiments_log.csv        # Saved experiment results
datasets/                  # Source datasets
split/
  ColBERT_dataset_split/   # Prepared train/validation/test files
  Kaggle_dataset_split/
test/
  predictions.csv          # Test predictions and class probabilities
```

**Installation**

The requirements specify a CUDA 12.1 build of PyTorch. Install it first:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements_minimal.txt
```


The script uses CUDA when available, otherwise CPU. Pretrained model files may be downloaded on the first run.

**Running**

Run from the project root:

```bash
python learning_model.py
```

The current configuration uses (but you are free to change it):

```python
MODEL_NAME = "roberta-base"
DATASET_FILE = "kaggle_dataset.csv"
SPLIT_FOLDER = "split/Kaggle_dataset_split"
CREATE_SPLITS = False
```

To use BERT, change `MODEL_NAME` to `"bert-base-uncased"`.

To use the other dataset, update both `DATASET_FILE` and `SPLIT_FOLDER`. Set `CREATE_SPLITS = True` to regenerate the splits from the source CSV.

**Evaluation and Outputs**

Test metrics are **Accuracy, Precision, Recall, and F1**. 

- `best_model.pth` — best model weights and optimizer state.
- `test/predictions.csv` — decoded texts, true labels, predicted labels, and class probabilities.
- `experiments_log.csv` — model settings and test metrics, appended after each run.

The checkpoint and prediction file are overwritten on subsequent runs.

**Recorded Results examples**

| Dataset | Model | Accuracy | F1 |
|---|---|---:|---:|
| ColBERT dataset | BERT | 0.9843 | 0.9843 |
| Kaggle dataset | BERT | 0.9797 | 0.9676 |
| ColBERT dataset | RoBERTa | 0.9516 | 0.9512 |
| Kaggle dataset | RoBERTa | 0.9851 | 0.9761 |

These are individual runs recorded in `experiments_log.csv`. Training randomness is not explicitly seeded yet.


**Acknowledgments**

The ColBERT dataset comes from the
[ColBERT project](https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection).

This repository fine-tunes standard BERT and RoBERTa classifiers.
