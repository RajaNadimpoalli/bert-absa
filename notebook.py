#!/usr/bin/env python
# coding: utf-8

# # Aspect-Based Sentiment Analysis using BERT
# This notebook fine-tunes a BERT model to classify sentiment based on specific aspects of Amazon reviews.

# In[ ]:


get_ipython().system('pip install transformers datasets scikit-learn wandb -q')


# In[ ]:


import pandas as pd
import torch
import random
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ## 1. Load and Preprocess the Dataset

# In[ ]:


dataset = load_dataset("amazon_polarity", split="train[:10000]")
df = pd.DataFrame(dataset)
df.rename(columns={"label": "sentiment", "content": "text"}, inplace=True)

aspects = ['battery', 'display', 'performance', 'price', 'design']
df["aspect"] = df["text"].apply(lambda x: random.choice(aspects))

neutral_idx = df.sample(frac=0.2, random_state=42).index
df.loc[neutral_idx, "sentiment"] = 1
df.loc[df["sentiment"] == 1, "sentiment"] = 2

df["input_text"] = df["aspect"] + ": " + df["text"]
df = df.sample(n=3000, random_state=42).reset_index(drop=True)


# In[ ]:


train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["input_text"].tolist(),
    df["sentiment"].tolist(),
    test_size=0.2,
    random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)


# ## 2. Define Dataset and Model

# In[ ]:


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)


# ## 3. Training Setup

# In[ ]:


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    do_eval=True,
    logging_steps=10
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.evaluate()


# ## 4. Prediction Function

# In[ ]:


def predict_sentiment(aspect, review_text):
    input_text = f"{aspect}: {review_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred_label]


# In[ ]:


examples = [
    ("battery", "The battery lasts all day and charges quickly."),
    ("display", "The screen resolution is terrible."),
    ("price", "For the price, this product is unbeatable."),
    ("performance", "It's fast and handles everything I throw at it."),
    ("design", "Looks cheap and feels flimsy.")
]

for aspect, review in examples:
    sentiment = predict_sentiment(aspect, review)
    print(f"Aspect: {aspect.ljust(12)} | Sentiment: {sentiment} | Review: {review}")

