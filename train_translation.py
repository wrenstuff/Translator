import os
import torch
import multiprocessing
from datasets import load_dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)

def main():
    # ====================================================
    # 0. CPU / Hardware Setup
    # ====================================================
    num_cpus = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    os.environ["MKL_NUM_THREADS"] = str(num_cpus)
    torch.set_num_threads(num_cpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    print(f"✅ Detected {num_cpus} CPU cores")

    # ====================================================
    # 1. Load Dataset
    # ====================================================
    dataset = load_dataset(
        'csv',
        data_files={
            'train': 'train.csv',
            'validation': 'val.csv',
            'test': 'test.csv'
        }
    )

    print("\n📦 Dataset loaded:")
    print(dataset)

    # ====================================================
    # 2. Model & Tokenizer
    # ====================================================
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    # Comment out section 2 and replace with below to upgrade a model
    # MODEL_PATH = ""  # the existing models path
    #
    # ====================================================
    # For Loading an EXISTING model + tokenizer
    # ====================================================
    # tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    # model = MarianMTModel.from_pretrained(MODEL_PATH)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    # ====================================================
    # 3. Preprocessing Function
    # ====================================================
    max_length = 128

    def preprocess_function(batch):
        inputs = batch["source"]
        targets = batch["target"]

        model_inputs = tokenizer(
            inputs, max_length=max_length, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_length, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\n🧩 Tokenizing dataset using all CPU cores...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_cpus,  # Parallel tokenization
        remove_columns=dataset["train"].column_names
    )
    print("✅ Tokenization complete")

    # ====================================================
    # 4. Data Collator
    # ====================================================
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ====================================================
    # 5. Training Arguments
    # ====================================================
    gpu_capability = torch.cuda.get_device_capability()
    supports_bf16 = gpu_capability[0] >= 8  # 8.x+ = BF16 capable
    supports_bf16 = False

    # Detect BF16 support
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        supports_bf16 = major >= 8

    generation_config = GenerationConfig(
        max_length=128,
        num_beams=4
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./translation_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        generation_config=generation_config,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        report_to="none",  # disables wandb/tensorboard
        load_best_model_at_end=True,
        dataloader_num_workers=max(1, num_cpus - 1),  # multi-core dataloading
    )

    # ====================================================
    # 6. Trainer
    # ====================================================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ====================================================
    # 7. Train & Save
    # ====================================================
    print("\n🚀 Starting training...")
    trainer.train()
    print("✅ Training complete")

    trainer.save_model("./final_translation_model")
    tokenizer.save_pretrained("./final_translation_model")
    print("💾 Model and tokenizer saved to ./final_translation_model")

    # ====================================================
    # 8. Evaluate on Test Set
    # ====================================================
    print("\n📊 Evaluating on test set...")
    results = trainer.evaluate(tokenized_datasets["test"])
    print("✅ Test Results:", results)

# ====================================================
# Windows multiprocessing entry point
# ====================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # prevents spawn issues on Windows
    main()
