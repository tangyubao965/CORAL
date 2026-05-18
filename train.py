import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from model import CoralModel
from optimizer import get_optimizer
from dataset import CoralDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    train_dataset = CoralDataset(args.train_file, tokenizer, args.max_input_length, args.max_target_length)
    dev_dataset = CoralDataset(args.dev_file, tokenizer, args.max_input_length, args.max_target_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    model = CoralModel(args.pretrained_model_path).to(args.device)
    optimizer = get_optimizer(model, args.learning_rate)

    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            view_type = batch['view_type'].to(args.device)

            loss = model(input_ids, attention_mask, labels, view_type)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

if __name__ == "__main__":
    main()