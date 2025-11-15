from transformers import MarianMTModel, MarianTokenizer
import torch
import os

MODEL = "german_model"

tokenizer = MarianTokenizer.from_pretrained(MODEL)
model = MarianMTModel.from_pretrained(MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text):
    encoded = tokenizer([text], return_tensors="pt", padding=True).to(device)
    generated = model.generate(**encoded)
    out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return out