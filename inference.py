import torch
import time
import os
import random
from datetime import datetime
from pathlib import Path
from nunchaku.models.safety_checker import SafetyChecker
from nunchaku.pipelines import flux as nunchaku_flux
from config import prompt_configs, PromptConfig

# Constants
MAX_SEED = 1000000000
OUTPUT_DIR = "output"

def generate_random_seed():
    return random.randint(0, MAX_SEED)

def save_image(image, prompt, seed):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    prompt_prefix = prompt.replace(' ', '_')[:15]
    prompt_prefix = ''.join(c for c in prompt_prefix if c.isalnum() or c == '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{prompt_prefix}_{timestamp}_{seed}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    return filepath

def initialize_pipeline():
    pipeline = nunchaku_flux.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        qmodel_path="mit-han-lab/svdquant-models/svdq-int4-flux.1-schnell.safetensors",
    )
    pipeline.enable_sequential_cpu_offload()
    return pipeline

def generate_image(pipeline, config: PromptConfig):
    seed = config.seed if config.seed is not None else generate_random_seed()
    
    start_time = time.time()
    image = pipeline(
        prompt=config.prompt,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    end_time = time.time()
    
    latency = end_time - start_time
    latency_str = f"{latency * 1000:.2f}ms" if latency < 1 else f"{latency:.2f}s"
    print(f"Generated image for prompt: '{config.prompt[:50]}...' - Latency: {latency_str}")
    
    return image, seed

def main():
    pipeline = initialize_pipeline()
    safety_checker = SafetyChecker("cuda", disabled=True)
    
    print(f"Starting generation for {len(prompt_configs)} prompts...")
    
    for idx, config in enumerate(prompt_configs, 1):
        print(f"\nProcessing prompt {idx}/{len(prompt_configs)}")
        image, seed = generate_image(pipeline, config)
        filepath = save_image(image, config.prompt, seed)
        print(f"Saved image to: {filepath}")

if __name__ == "__main__":
    main()
