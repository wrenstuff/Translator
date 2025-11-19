from transformers import MarianMTModel, MarianTokenizer
import torch

# =========================
# Model Loader
# =========================
def load_model(model_name):
    """Ensures that the model is loaded correctly. Using model path as input"""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Ensure bos_token_id exists
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"

    if tokenizer.bos_token_id is None:
        tokenizer.add_tokens(["<s>"], special_tokens=True)

    # Set decoder_start_token_id correctly
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id

    return model, tokenizer


es_model, es_tokenizer = load_model("spanish_model")
de_model, de_tokenizer = load_model("german_model")


# =========================
# Text Translator Function
# =========================
def translate_text(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer([text], return_tensors="pt", padding=True).to(device)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    generated = model.generate(**encoded, decoder_start_token_id=model.config.decoder_start_token_id)
    out = tokenizer.batch_decode(generated[0], skip_special_tokens=True)
    translation = " ".join(out).strip()
    return translation
