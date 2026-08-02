import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from opacus.accountants import RDPAccountant

@torch.no_grad()
def compute_per_example_losses(model, tokenizer, records, task_type, label_set=None, max_examples=None):
    model.eval()
    losses = []
    recs = records[:max_examples] if max_examples else records
    for context, target in recs:
        if task_type == "lm":
            prompt, target_text = context, context
        elif task_type == "classification":
            prompt, target_text = build_prompt(task_type, context, label_set), target
        else:
            prompt, target_text = build_prompt(task_type, context), target
        ids, labels = make_supervised_example(tokenizer, prompt, target_text, task_type)
        ids_t, lab_t = ids.unsqueeze(0).to(model.device), labels.unsqueeze(0).to(model.device)
        out = model(input_ids=ids_t, labels=lab_t)
        losses.append(out.loss.item())
    return np.array(losses)

def run_loss_based_mia(model, tokenizer, member_records, nonmember_records, task_type,
                        label_set=None, max_examples=None):
    """Simple loss-based membership inference attack (Yeom et al., 2018): lower loss on
    an example is treated as evidence the model was trained on it. Returns the attack's
    ROC-AUC - 0.5 means chance level (good: membership isn't inferable from loss alone),
    materially above 0.5 means some membership signal leaked through.
    """
    member_losses = compute_per_example_losses(model, tokenizer, member_records, task_type,
                                                  label_set, max_examples)
    nonmember_losses = compute_per_example_losses(model, tokenizer, nonmember_records, task_type,
                                                     label_set, max_examples)
    if len(member_losses) == 0 or len(nonmember_losses) == 0:
        return float("nan")
    scores = np.concatenate([-member_losses, -nonmember_losses])  # lower loss -> higher "is member" score
    labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(nonmember_losses)])
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))