import os
import subprocess
from loguru import logger

def install_packages_for_inference():
    commands = [
        "pip install --upgrade pip",
        "pip install --upgrade torch",
        "pip install --upgrade accelerate transformers",
        "pip uninstall transformer-engine -y",
    ]
    try:
        for cmd in commands:
            logger.info(f"Running command: {cmd}")
            res = subprocess.run(cmd, check=True, shell=True, capture_output=True)
            if res.returncode != 0:
                logger.error(f"Error: {res.stderr.decode('utf-8')}")
                return res.stderr.decode("utf-8")
        return None
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

def run_inference(model):
    import time
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch
    import pandas as pd
    from prompts import list_of_questions

    number_of_prompts = int(os.environ.get("INFERENCE_PROMPTS_SIZE", 30))
    list_of_questions = list_of_questions[:number_of_prompts]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        start_time = time.time()
        for prompt in list_of_questions:
            sequences = pipeline(
                prompt,
                max_length=1000,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            for seq in sequences:
                print(f"Result: {seq['generated_text']}")
        end_time = time.time()

        return end_time - start_time, None
    except Exception as e:
        print(f"Error: {e}")
        return 0, f"Error: {e}"