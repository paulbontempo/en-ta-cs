import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import json

# Configuration
MODEL_NAME = "xlm-roberta-base"
TRAIN_FILE = "train.conll"  # Your labeled training data
VAL_FILE = "val.conll"      # Validation data (optional, can split from train)
OUTPUT_DIR = "xlmr-langid-model"
MAX_LENGTH = 128

# Label mappings
LABEL_LIST = ["en", "ta", "na"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def read_conll_file(file_path):
    """Read CoNLL format file and return sentences with labels."""
    sentences = []
    labels = []
    
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Empty line = end of sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word, label = parts
                    current_sentence.append(word)
                    current_labels.append(label)
        
        # Don't forget the last sentence
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    return sentences, labels


def create_dataset(sentences, labels, tokenizer):
    """Convert sentences and labels to dataset format."""
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH
    )
    
    # Align labels with tokenized inputs
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                label_ids.append(LABEL2ID[label[word_idx]])
            else:
                # Subsequent subwords of the same word get -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        aligned_labels.append(label_ids)
    
    tokenized_inputs["labels"] = aligned_labels
    return Dataset.from_dict(tokenized_inputs)


def compute_metrics(pred):
    """Compute accuracy and per-class metrics."""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_labels = [[ID2LABEL[l] for l in label if l != -100] 
                   for label in labels]
    true_predictions = [[ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    
    # Flatten for sklearn metrics
    flat_true = [label for sublist in true_labels for label in sublist]
    flat_pred = [pred for sublist in true_predictions for pred in sublist]
    
    # Calculate metrics
    accuracy = accuracy_score(flat_true, flat_pred)
    report = classification_report(flat_true, flat_pred, 
                                   target_names=LABEL_LIST, 
                                   output_dict=True)
    
    return {
        "accuracy": accuracy,
        "en_f1": report["en"]["f1-score"],
        "ta_f1": report["ta"]["f1-score"],
        "na_f1": report["na"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"]
    }


def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    print("Loading training data...")
    train_sentences, train_labels = read_conll_file(TRAIN_FILE)
    print(f"Loaded {len(train_sentences)} training sentences")
    
    # Load validation data if exists, otherwise split from training
    try:
        val_sentences, val_labels = read_conll_file(VAL_FILE)
        print(f"Loaded {len(val_sentences)} validation sentences")
    except FileNotFoundError:
        print("No validation file found, using 10% of training data for validation")
        split_idx = int(len(train_sentences) * 0.9)
        val_sentences = train_sentences[split_idx:]
        val_labels = train_labels[split_idx:]
        train_sentences = train_sentences[:split_idx]
        train_labels = train_labels[:split_idx]
        print(f"Split: {len(train_sentences)} train, {len(val_sentences)} validation")
    
    print("Creating datasets...")
    train_dataset = create_dataset(train_sentences, train_labels, tokenizer)
    val_dataset = create_dataset(val_sentences, val_labels, tokenizer)
    
    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none"  # Disable wandb
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nEvaluating on validation set...")
    results = trainer.evaluate()
    print(f"\nValidation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save label mappings
    with open(f"{OUTPUT_DIR}/label_config.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
    
    print("Training complete!")


if __name__ == "__main__":
    main()