#!/usr/bin/env python
"""
Generate adversarial prompts with AdvLLM, then (after freeing that model)
score them for perplexity and write everything to one JSON file.

Output file:
    {store_path}/{dataset}/{mode|group_beam_search_N}/{advllm}.json

Content:
    • greedy / do_sample
        {
          "adv_prompts":              ["prompt‑0", "prompt‑1", …],
          "perplexities":             [23.1, 19.4, …],
          "adv_prompts_goal_rep4":    ["g g g prompt‑0", …],   # (repeat‑1)×goal + prompt
          "perplexities_goal_rep4":   [27.8, 24.2, …]
        }

    • group_beam_search   (num_beams = N)
        {
          "adv_prompts":              [["p0_b0", …, "p0_bN‑1"],
                                       ["p1_b0", …, "p1_bN‑1"],
                                       …],
          "perplexities":             [[17.3, …], …],
          "adv_prompts_goal_rep4":    [["g0 g0 g0 p0_b0", …], …],
          "perplexities_goal_rep4":   [[20.9, …], …]
        }
"""

import os, json, argparse, numpy as np, torch, evaluate
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
from .string_utils import get_goals_and_targets

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seeds(seed: int = 20):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_generation_config(mode: str, max_tokens: int, num_beams: int = 1) -> GenerationConfig:
    if mode == "greedy":
        return GenerationConfig(do_sample=False, max_new_tokens=max_tokens)
    if mode == "do_sample":
        return GenerationConfig(do_sample=True, temperature=0.6, top_p=0.9,
                                max_new_tokens=max_tokens)
    if mode == "group_beam_search":
        return GenerationConfig(do_sample=False,
                                diversity_penalty=1.0,
                                num_beams=num_beams,
                                num_beam_groups=num_beams,
                                num_return_sequences=num_beams,
                                max_new_tokens=max_tokens,
                                min_new_tokens=max_tokens)
    raise ValueError(f"Unknown mode '{mode}'")

def build_store_path(base: str, dataset: str, mode: str, num_beams: int) -> str:
    suffix = f"group_beam_search_{num_beams}" if mode == "group_beam_search" else mode
    path = os.path.join(base, dataset, suffix)
    os.makedirs(path, exist_ok=True)
    return path

def prepend_goal(goal: str, adv: str, repeat: int) -> str:
    """Return `goal` prefixed (repeat‑1) times to `adv`."""
    return " ".join([goal]*(repeat-1) + [adv])

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.set_verbosity_error()
    set_seeds()

    parser = argparse.ArgumentParser("Adv prompt generation + perplexity")
    # Decoding strategy
    parser.add_argument("--mode", default="greedy",
                        choices=["greedy", "do_sample", "group_beam_search"])
    parser.add_argument("--num_beams", type=int, default=50)

    # Models / paths
    parser.add_argument("--advllm", default="cesun/advllm_llama3")
    parser.add_argument("--ppl_model",  default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--store_path", default="generated_adv_prompts/")

    # Dataset
    parser.add_argument("--dataset", default="advbench", choices=["advbench", "mlcinst"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--n_train_data", type=int, default=520)

    # Perplexity options
    parser.add_argument("--ppl_batch", type=int, default=4)

    # Goal‑repeat
    parser.add_argument("--repeat", type=int, default=4,
                        help="Total goal occurrences wanted (original prompt already has 1).")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------- load AdvLLM ------------------------------ #
    adv_model = (AutoModelForCausalLM
                 .from_pretrained(args.advllm, torch_dtype=torch.bfloat16)
                 .to(device).eval())
    adv_tok = AutoTokenizer.from_pretrained(args.advllm,
                                            trust_remote_code=True,
                                            use_fast=False)
    adv_tok.pad_token = adv_tok.eos_token

    # ---------------------------- generation cfg --------------------------- #
    max_tokens = 70 if "llama3" in args.advllm.lower() else 90
    gen_cfg    = build_generation_config(args.mode, max_tokens, args.num_beams)

    # ---------------------------- load goals ------------------------------- #
    if args.dataset == "advbench":
        goals, _ = get_goals_and_targets("data/advbench/harmful_behaviors.csv",
                                         args.offset, args.n_train_data)
    else:
        with open("data/MaliciousInstruct.txt", encoding="utf-8") as f:
            goals = [line.strip() for line in f]

    # ---------------------------- generate prompts ------------------------- #
    adv_prompts_flat: List[str] = []
    rep4_prompts_flat: List[str] = []

    if args.mode == "group_beam_search":
        adv_prompts_2d: List[List[str]] = []
        rep4_prompts_2d: List[List[str]] = []

    for goal in goals:
        inputs  = adv_tok(goal, return_tensors="pt").to(device)
        outputs = adv_model.generate(**inputs, generation_config=gen_cfg)

        if args.mode == "group_beam_search":
            decoded = adv_tok.batch_decode(outputs, skip_special_tokens=True)
            adv_prompts_2d.append(decoded)
            adv_prompts_flat.extend(decoded)

            rep4_list = [prepend_goal(goal, adv, args.repeat) for adv in decoded]
            rep4_prompts_2d.append(rep4_list)
            rep4_prompts_flat.extend(rep4_list)
        else:
            prompt = adv_tok.decode(outputs[0], skip_special_tokens=True)
            adv_prompts_flat.append(prompt)

            rep4_prompt = prepend_goal(goal, prompt, args.repeat)
            rep4_prompts_flat.append(rep4_prompt)

    # ---------------------------- free AdvLLM ------------------------------ #
    del adv_model, adv_tok
    torch.cuda.empty_cache()

    # ---------------------------- perplexity ------------------------------- #
    ppl_metric = evaluate.load("perplexity", module_type="metric")

    # first the original prompts
    ppl_metric.add_batch(predictions=adv_prompts_flat)
    ppl_adv_flat = ppl_metric.compute(model_id=args.ppl_model,
                                      max_length=100,
                                      batch_size=args.ppl_batch)["perplexities"]
    ppl_metric._batch = []

    # then the rep4 prompts
    ppl_metric.add_batch(predictions=rep4_prompts_flat)
    ppl_rep4_flat = ppl_metric.compute(model_id=args.ppl_model,
                                       max_length=100,
                                       batch_size=args.ppl_batch)["perplexities"]

    # ---------------------------- reshape / package ------------------------ #
    if args.mode == "group_beam_search":
        # reshape both PPL lists to 2‑D
        nb = args.num_beams
        ppl_adv_2d  = [ppl_adv_flat[i*nb:(i+1)*nb]   for i in range(len(goals))]
        ppl_rep4_2d = [ppl_rep4_flat[i*nb:(i+1)*nb]  for i in range(len(goals))]

        out_dict = {
            "adv_prompts":              adv_prompts_2d,
            "perplexities":             ppl_adv_2d,
            "adv_prompts_goal_rep4":    rep4_prompts_2d,
            "perplexities_goal_rep4":   ppl_rep4_2d
        }
    else:
        out_dict = {
            "adv_prompts":              adv_prompts_flat,
            "perplexities":             ppl_adv_flat,
            "adv_prompts_goal_rep4":    rep4_prompts_flat,
            "perplexities_goal_rep4":   ppl_rep4_flat
        }

    # ---------------------------- save ------------------------------------- #
    store_dir = build_store_path(args.store_path, args.dataset,
                                 args.mode, args.num_beams)
    out_file  = os.path.join(store_dir,
                             f"{args.advllm.replace('/','_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)

    print(f"Saved output to {out_file}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
