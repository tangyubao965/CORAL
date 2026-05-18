
config = {
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    "tokenizer_name": "t5-base",
    "model_name": "t5-base",
    "batch_size": 8,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_epochs": 3,
    "max_length": 128,
    "save_path": "coral_model.pt"
}
