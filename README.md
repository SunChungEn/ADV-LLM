# ADV-LLM

**2025/5/29 Update:** The full code and models are now updated! Sorry for the delay, as we needed to undergo code review for jailbreak-related works.

**2025/4/15 update:** This work has passed the code review from Microsoft. We will release the code and models in a few weeks.

**2025/1/22 Update:** Our paper has been accepted to NAACL 2025 main conference!


This is the official repository for the paper: [**Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**](https://arxiv.org/abs/2410.18469). Work done during an internship at Microsoft Research.

This code is based on [GCG attack](https://github.com/llm-attacks/llm-attacks) and [BEAST](https://github.com/vinusankars/BEAST).

## Overview
<p align="center">
  <img src="./fig/advllm.png" width="80%" height="80%" />
</p>

## Installation

### Method 1: Quick Install (Recommended)
Install the package directly from the repository:
```bash
pip install -e .
```

This will install all required dependencies and make the `advllm` command available in your environment.

### Method 2: Manual Setup
We recommend using CUDA 12.1, Python 3.10, and PyTorch 2.2.

Install the required packages:
```bash
pip install -r requirements.txt
```

## ADV-LLMs on Hugging Face
We have provided all the ADV-LLMs on Hugging Face:

- [cesun/advllm_llama3](https://huggingface.co/cesun/advllm_llama3)
- [cesun/advllm_llama2](https://huggingface.co/cesun/advllm_llama2)
- [cesun/advllm_mistral](https://huggingface.co/cesun/advllm_mistral)
- [cesun/advllm_guanaco](https://huggingface.co/cesun/advllm_guanaco)
- [cesun/advllm_vicuna](https://huggingface.co/cesun/advllm_vicuna)
- [cesun/advllm_phi3](https://huggingface.co/cesun/advllm_phi3)

`advllm_llama3` means it is based on LLaMA3-8B and is self-tuned for attacking LLaMA3.

## (Optional) Train ADV-LLMs (Iterative Finetuning Algorithm)
This step requires 8 A100 GPUs (80GB RAM). If there are not enough GPUs, please modify the arguments in `advllm_selftuning.sh`. For example, to train with 4 A100 GPUs, change `--num_processes 8` to `--num_processes 4`.

To train an ADV-LLM with the target being Vicuna v1.5, run:
```bash
advllm_train 0 5 vicuna
```
The ADV-LLM starts from the same instruction-tuned model as the target. The default iteration range is from 0 (inclusive) to 5 (exclusive).

Change the model name to `guanaco`, `mistral`, `phi3`, `llama2`, or `llama3` for other models.

The checkpoint is saved at each iteration. If the process accidentally stops after iteration 2, you can resume by running:
```bash
advllm_train 3 5 vicuna
```

## (Optional) Generate Adversarial Prompts with ADV-LLMs
In this step, we use the ADV-LLMs from Huggingface directly to generate adversarial prompts. Training ADV-LLMs locally is optional.

To generate adversarial prompts with `advllm_llama3`, run:
```bash
advllm_get_adv_prompts --advllm cesun/advllm_llama3
```

Change `--advllm` to your custom path if you use other ADV-LLMs. Set the decoding mode `--mode` to `greedy`, `do_sample`, or `group_beam_search`. Our paper reports results using `greedy` and `group_beam_search`. For `group_beam_search`, it generates 50 adversarial suffixes for each harmful query (GBS50).

Results, including adversarial prompts and perplexities, are saved to a JSON file. To switch the dataset to **MaliciousInstruct**, add:
```bash
--dataset mlcinst
```

We have also provided pre-generated adversarial prompts (saved under `generated_adv_prompts/`), so you can skip this step. However, these are generated with GBS50 decoding only. To evaluate under other settings, please generate prompts yourself.

## Evaluate the ASR
This step requires exporting an OpenAI API key, as GPT-4 is used to judge whether an attack is successful. It is also needed when attacking GPT-series models.

Three metrics for evaluating ASR:
- **Template Check**
- **Llama Guard Check**
- **GPT-4 Check**

Final results are stored under `attack_results/` like:
```
Total ASR (template): 1.0
Total ASR (llama guard): 1.0
Total ASR (gpt4): 1.0
```
Including full logs and all successful examples in separate files.

### Evaluate Open Source Models
To evaluate the ASR from ADV-LLM (optimized on LLaMA3) using **greedy decoding** on **advbench** dataset:
```bash
advllm_eval --advllm cesun/advllm_llama3 --target_model llama3
```
This will attack the LLaMA3 model.

For transfer attacks (between models), change `--target_model`, for example:
```bash
--target_model vicuna
```

Evaluate only the first 100 queries by adding:
```bash
--n_train_data 100
```

### Evaluate with Group Beam Search
To evaluate under GBS50 settings, add:
```bash
--mode group_beam_search
```

### Evaluate Closed Source Models
To attack GPT-3.5-turbo or GPT-4-turbo from ADV-LLM (optimized on LLaMA3), set:
```bash
--target_model GPT-3.5-turbo
```
To switch to GPT-4:
```bash
--target_model gpt-4-turbo
```

### Evaluate with OOD Queries
Switch the dataset to MaliciousInstruct (for OOD test) with:
```bash
--dataset mlcinst
```
### Evaluate under Perplexity Defense
To test the ASR under perplexity defense, simply add:
```bash
--perplexity_defense
```
This checks each adversarial prompt with a perplexity filter.

To enable repetition of the harmful query to further lower perplexity:
```bash
--rep4
```

## Cite this Work
Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao, "Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities"
```bibtex
@article{advllm,
   title={Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities},
   author={Chung-En Sun and Xiaodong Liu and Weiwei Yang and Tsui-Wei Weng and Hao Cheng and Aidan San and Michel Galley and Jianfeng Gao},
   journal={NAACL},
   year={2025}
}
```
