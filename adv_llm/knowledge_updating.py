import gc
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import time
import json

from string_utils import model_path_dict


from datasets import Dataset

# np.random.seed(20)
# torch.manual_seed(20)
# torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vicuna")
parser.add_argument("--target_models", type=str, default="vicuna")
parser.add_argument("--current_iteration", type=int, default=0)
parser.add_argument("--store_path", type=str, default="advllm_models/")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

store_path = args.store_path + args.target_models + "/"
if not os.path.exists(store_path):
    os.makedirs(store_path)


if args.current_iteration == 0:
    model_path = model_path_dict[args.model]
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
else:
    model_path = model_path_dict[args.model]
    model = AutoModelForCausalLM.from_pretrained(store_path + f"advllm_{args.model}_{args.current_iteration-1}", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.do_sample = True

data = []
for i in range(args.current_iteration + 1):
    with open(store_path + f"sampled_suffixes_{i}.json", 'r', encoding='utf-8') as file:
        d = json.load(file)
        data.extend(d)

filtered_data = []
for i in range(len(data)):
    if not 0 in data[i]['jailbroken']:
        filtered_data.append(data[i])
print("total_data:", len(data))
print("success_data:", len(filtered_data))

dataset = Dataset.from_list(filtered_data)
dataset = dataset.map(lambda e: tokenizer(e['adv_prompt'], truncation=True, padding=True, return_tensors="pt"), batched=True, batch_size=len(dataset)).remove_columns(['adv_prompt'])
dataset.set_format(type='torch', columns=['jailbroken', 'suffix_len', 'loss', 'input_ids', 'attention_mask', 'train_slice'])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


for e in range(args.epochs):
    model.train()
    training_loss = []
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        mask = []
        for s in batch['train_slice']:
            m = torch.zeros(batch['input_ids'].shape[-1])
            m[s[0]: s[1]] = 1.0
            mask.append(m)
        mask = torch.stack(mask, dim=0).to(batch['input_ids'].device)

        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
        loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), batch['input_ids'][:, 1:].reshape(-1)) * mask[:, 1:].reshape(-1)
        loss = loss.reshape(batch['input_ids'].shape[0], -1)
        loss = loss.sum(dim=-1) / mask.sum(dim=-1)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("batch ", str(i), " loss: ", loss.detach().cpu().numpy(), end="\r")
        training_loss.append(loss.detach().cpu().numpy())
    avg_training_loss = sum(training_loss)/len(training_loss)
    print("training loss: ", avg_training_loss)

model.save_pretrained(store_path + f"advllm_{args.model}_{args.current_iteration}")