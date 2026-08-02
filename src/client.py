import random
import numpy as np
import torch
import math

def partition_iid(records, num_clients, seed=42):
    rng = random.Random(seed)
    recs = records[:]
    rng.shuffle(recs)
    shards = [[] for _ in range(num_clients)]
    for i, r in enumerate(recs):
        shards[i % num_clients].append(r)
    return shards

def partition_dirichlet(records, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    label_to_idx = {}
    for i, (_, lab) in enumerate(records):
        label_to_idx.setdefault(lab, []).append(i)
    shards = [[] for _ in range(num_clients)]
    for lab, idxs in label_to_idx.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        for c, split in enumerate(np.split(idxs, split_points)):
            for i in split:
                shards[c].append(records[i])
    return shards

def partition_dataset(records, task_type, num_clients=5):
    if task_type == "classification":
        return partition_dirichlet(records, num_clients)
    return partition_iid(records, num_clients)

def make_supervised_example(tokenizer, prompt_text, target_text, task_type, max_len=256):
    if task_type == "lm":
        enc = tokenizer(prompt_text, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
    else:
        target = " " + str(target_text)
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len - 8,
                                 return_tensors="pt")["input_ids"][0]
        target_ids = tokenizer(target, truncation=True, max_length=8, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
        input_ids = torch.cat([prompt_ids, target_ids])[:max_len]
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids])[:max_len]
    return input_ids, labels

def collate_batch(examples, pad_id):
    max_len = max(len(x[0]) for x in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        input_ids[i, :len(ids)] = ids
        labels[i, :len(lab)] = lab
        attn[i, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

def train_one_round_dp(model, tokenizer, client_records, task_type, round_idx, dp_enabled=True, 
                       client_samples_per_round=100, train_batch_size=1, dp_max_grad_norm=1.0, dp_noise_multiplier=1.0):
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    n = max(1, len(client_records))
    start_idx = (round_idx * client_samples_per_round) % n
    batch_recs = (client_records * 2)[start_idx:start_idx + client_samples_per_round]
    if not batch_recs:
        return float("nan")

    if not dp_enabled:
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
        examples = []
        for context, target in batch_recs:
            prompt = build_prompt(task_type, context)
            target_text = context if task_type == "lm" else target
            ids, labels = make_supervised_example(tokenizer, prompt, target_text, task_type)
            examples.append((ids, labels))
        pad_id = tokenizer.pad_token_id
        losses = []
        for i in range(0, len(examples), train_batch_size):
            batch = examples[i:i + train_batch_size]
            batch_t = collate_batch(batch, pad_id)
            batch_t = {k: v.to(model.device) for k, v in batch_t.items()}
            optimizer.zero_grad()
            out = model(**batch_t)
            out.loss.backward()
            losses.append(out.loss.item())
            optimizer.step()
        return float(np.mean(losses)) if losses else float("nan")

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
    accum = [torch.zeros_like(p) for p in trainable_params]
    losses = []

    for context, target in batch_recs:
        prompt = build_prompt(task_type, context)
        target_text = context if task_type == "lm" else target
        ids, labels = make_supervised_example(tokenizer, prompt, target_text, task_type)
        ids, labels = ids.unsqueeze(0).to(model.device), labels.unsqueeze(0).to(model.device)

        optimizer.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        losses.append(out.loss.item())

        total_norm_sq = sum((p.grad.detach() ** 2).sum() for p in trainable_params if p.grad is not None)
        total_norm = float(total_norm_sq.sqrt().item())
        clip_coef = min(1.0, dp_max_grad_norm / (total_norm + 1e-6))
        for a, p in zip(accum, trainable_params):
            if p.grad is not None:
                a.add_(p.grad.detach() * clip_coef)

    num_examples = len(batch_recs)
    for a, p in zip(accum, trainable_params):
        noise = torch.normal(mean=0.0, std=dp_noise_multiplier * dp_max_grad_norm,
                              size=a.shape, device=a.device).to(a.dtype)
        p.grad = (a + noise) / num_examples

    optimizer.step()
    optimizer.zero_grad()

    return float(np.mean(losses)) if losses else float("nan")