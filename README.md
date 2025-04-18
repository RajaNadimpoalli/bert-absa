# Aspect-Based Sentiment Analysis using BERT

This project implements and evaluates a fine-tuned BERT model for Aspect-Based Sentiment Analysis (ABSA) using a simulated version of the Amazon Polarity dataset. It classifies review sentiments based on specific product aspects such as battery, design, or performance.

---

## 📌 Project Overview

- **Model**: `bert-base-uncased` fine-tuned for 3-class sentiment classification (Negative, Neutral, Positive)
- **Architecture**: BERT encoder with a linear classifier on [CLS] token
- **Data**: Amazon Polarity dataset (subset of 3,000 reviews)
- **Tracking**: Training progress visualized using [Weights & Biases](https://wandb.ai/)
- **Platform**: Developed and tested on Google Colab (T4 GPU)

---

## 📁 Project Structure

| File/Folder         | Description |
|---------------------|-------------|
| `notebook.ipynb`    | Full training workflow |
| `predict.py`        | Inference script for making predictions |
| `requirements.txt`  | Python dependencies |
| `results/`          | Training outputs and model logs |
| `BERT_ABSA_Project_Presentation.pptx` | Final project presentation |
| `README.md`         | You're reading it! |

---

## 🧠 Model Training

- Input format: `[aspect]: review text`
- Tokenizer: `bert-base-uncased` (max length = 128)
- Optimizer: AdamW (learning rate = 5e-5)
- Batch size: 8, Epochs: 3
- Loss: Cross-entropy
- Metrics: Accuracy and weighted F1-score

---

## 📊 Results

| Metric              | Value     |
|---------------------|-----------|
| Validation Accuracy | ~80%      |
| F1 Score (weighted) | ~0.78     |
| Evaluation Loss     | 0.44      |

Example prediction format:
```python
Aspect: battery | Sentiment: Positive | Review: "The battery lasts all day and charges quickly."
## 📁 Notebooks & Outputs

| File | Description |
|------|-------------|
| [`notebook.ipynb`](./notebook.ipynb) | Original notebook file (may not render on GitHub) |
| [`notebook.py`](./notebook.py) | Converted Python source version (fully functional) |
| [`notebook.pdf`](./notebook.pdf) | Rendered PDF version with training results and predictions |

