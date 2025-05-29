#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, logfile=None):
    """Run a command and optionally log its output."""
    try:
        if logfile:
            with open(logfile, 'a') as f:
                process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        else:
            process = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial training for large language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported model names: vicuna, guanaco, mistral, phi3, llama2, llama3

Example:
    advllm train --start 0 --end 5 --model vicuna
    
Note: This tool is designed for 8 A100 GPUs (80GB).
If you have fewer GPUs, use --num-gpus to specify the number of GPUs.
        """
    )
    
    parser.add_argument('--start', type=int, required=True,
                        help='Starting iteration index')
    parser.add_argument('--end', type=int, required=True,
                        help='Ending iteration index (exclusive)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., vicuna, guanaco, mistral, etc.)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use (default: 8)')
    
    args = parser.parse_args()
    
    if args.start >= args.end:
        print("Error: start index must be less than end index", file=sys.stderr)
        sys.exit(1)
    
    for i in range(args.start, args.end):
        logfile = f"advllm_{args.model}_{i}.log"
        print(f"=== Iteration {i} | Model: {args.model} ===")
        
        # Run suffix sampling
        cmd = [
            "accelerate", "launch",
            f"--num_processes", str(args.num_gpus),
            "adv_llm/suffix_sampling.py",
            "--current_iteration", str(i),
            "--model", args.model,
            "--target_models", args.model
        ]
        if not run_command(cmd, logfile):
            sys.exit(1)
        
        # Run knowledge updating
        cmd = [
            "python", "adv_llm/knowledge_updating.py",
            "--current_iteration", str(i),
            "--model", args.model,
            "--target_models", args.model
        ]
        if not run_command(cmd, logfile):
            sys.exit(1)

if __name__ == '__main__':
    main() 