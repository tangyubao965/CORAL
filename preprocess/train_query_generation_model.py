
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import os

def load_data(train_file, val_file):
    train_dataset = load_dataset("json", data_files={"train": train_file})["train"]
    val_dataset = load_dataset("json", data_files={"validation": val_file})["validation"]
    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=32):
    model_inputs = tokenizer(examples["input"], max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model(train_file, val_file, output_dir):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    dataset = load_data(train_file, val_file)
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    train_model(args.train_file, args.val_file, args.output_dir)
