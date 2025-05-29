import copy
import torch
import math
import numpy as np
from transformers import (GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)

# This function is based on BEAST implementation: https://github.com/vinusankars/BEAST
# "Fast Adversarial Attacks on Language Models In One GPU Minute" by Vinu Sankar Sadasivan, Shoumik Saha, Gaurang Sriramanan, Priyatham Kattakinda, Atoosa Chegini, and Soheil Feizi.
@torch.no_grad()
def generate_n_tokens_batch(model, ids, att_mask, max_gen_len, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0):
    assert max(len(i) for i in ids) == min(len(i) for i in ids), "Need to pad the batch"        
    if max_gen_len == 0:
        return None, ids
    
    config = copy.deepcopy(model.generation_config)
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p
    model.generation_config.top_k = top_k
    model.generation_config.repetition_penalty = repetition_penalty
    
    prompt_len = len(ids[0])
    kwargs = copy.deepcopy({"generation_config": model.generation_config, "max_length": max_gen_len+prompt_len, \
        "min_length": max_gen_len+prompt_len, "return_dict_in_generate": True, "output_scores": True})
    out = model.generate(input_ids=ids, attention_mask=att_mask, **kwargs)

    tokens = out.sequences
    logits = torch.stack(out.scores)
    logits = torch.permute(logits, (1, 0, 2))

    model.generation_config = config
    return logits, tokens

# This function is based on BEAST implementation: https://github.com/vinusankars/BEAST
# "Fast Adversarial Attacks on Language Models In One GPU Minute" by Vinu Sankar Sadasivan, Shoumik Saha, Gaurang Sriramanan, Priyatham Kattakinda, Atoosa Chegini, and Soheil Feizi.
@torch.no_grad()
def sample(probs, return_tokens=0, not_allowed_tokens=None):
    
    if not_allowed_tokens is not None:
        probs[:, not_allowed_tokens] = 0.0
        # if (probs.sum(dim=-1) == 0.0).any():
        #     probs[torch.nonzero(probs.sum(dim=-1) == 0.0, as_tuple=True)[0], :] = 1.0 / probs.shape[-1]
        #     probs[:, not_allowed_tokens] = 0.0
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)   
    next_tokens = torch.multinomial(probs_sort, num_samples=max(1, return_tokens))
    next_tokens = torch.gather(probs_idx, -1, next_tokens)
    
    return next_tokens

@torch.no_grad()
def calculate_loss(model, ids, att_mask, target_slices, max_inference_batch=64):
    mask = []
    for s in target_slices:
        m = torch.zeros(ids.shape[-1])
        m[s[0]: s[1]] = 1.0
        mask.append(m)
    mask = torch.stack(mask, dim=0).to(ids.device)

    total_samples = ids.shape[0]
    ids = [ids[i : i + max_inference_batch] for i in range(0, len(ids), max_inference_batch)]
    att_mask = [att_mask[i : i + max_inference_batch] for i in range(0, len(att_mask), max_inference_batch)]
    mask = [mask[i : i + max_inference_batch] for i in range(0, len(mask), max_inference_batch)]
    all_losses = []
    for i in range(int(math.ceil(total_samples/max_inference_batch))):
        outputs = model(input_ids=ids[i], attention_mask=att_mask[i]).logits
        loss = torch.nn.CrossEntropyLoss(reduction='none')(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), ids[i][:, 1:].reshape(-1)) * mask[i][:, 1:].reshape(-1)
        loss = loss.reshape(ids[i].shape[0], -1)
        all_losses.append(loss.sum(dim=-1) / mask[i].sum(dim=-1))
    all_losses = torch.cat(all_losses, dim=0)

    return all_losses
        
@torch.no_grad()
def decay(x):
    return 2.3 * np.exp(-0.5 * x) + 0.7

# This function is based on GCG attack implementation: https://github.com/llm-attacks/llm-attacks
# "Universal and Transferable Adversarial Attacks on Aligned Language Models" by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson
@torch.no_grad()
def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

# This function is based on GCG attack implementation: https://github.com/llm-attacks/llm-attacks
# "Universal and Transferable Adversarial Attacks on Aligned Language Models" by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson
@torch.no_grad()
def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
# This function is based on GCG attack implementation: https://github.com/llm-attacks/llm-attacks
# "Universal and Transferable Adversarial Attacks on Aligned Language Models" by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson
def token_gradients(model, embed_weights, input_ids, adv_slice, target_slice, not_allowed_tokens=None):

    one_hot = torch.zeros(
        input_ids[adv_slice[1]-1: adv_slice[1]].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[adv_slice[1]-1: adv_slice[1]].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids[: target_slice[1]].unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:adv_slice[1]-1,:], 
            input_embeds, 
            embeds[:,adv_slice[1]:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice[0]: target_slice[1]]
    loss = torch.nn.CrossEntropyLoss()(logits[0,target_slice[0]-1: target_slice[1]-1,:], targets)
    
    loss.backward()
    
    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens] = np.infty
    
    return grad

