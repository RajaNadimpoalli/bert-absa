import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
MODEL_PATH = "./results"

def load_model_and_tokenizer(model_path=MODEL_PATH):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_sentiment(aspect, review, model, tokenizer, device):
    input_text = f"{aspect}: {review}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    
    return LABEL_MAP[pred]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", type=str, required=True)
    parser.add_argument("--review", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer()
    sentiment = predict_sentiment(args.aspect, args.review, model, tokenizer, device)
    print(f"\n🧠 Predicted Sentiment: {sentiment}")
