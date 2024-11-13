from typing import List, Dict

class PromptConfig:
    def __init__(self, prompt: str, width: int = 1024, height: int = 1024, 
                 num_inference_steps: int = 4, guidance_scale: float = 0, seed: int = None):
        self.prompt = prompt
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

# List of prompt configurations
prompt_configs: List[PromptConfig] = [
    PromptConfig(
        prompt="A misty Japanese garden at dawn, with a red maple tree dropping leaves onto a koi pond, cinematic lighting, photorealistic style",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Cyberpunk café interior, neon signs reflecting off rain-soaked windows, steam rising from coffee cups, volumetric lighting, highly detailed",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Ancient library with floating books and magical dust particles, rays of sunlight streaming through stained glass windows, fantasy art style",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Crystal cave with bioluminescent fungi, underground lake reflecting turquoise crystals, ethereal atmosphere, hyper-realistic detail",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Steampunk airship floating through golden sunset clouds, brass and copper details, steam trailing behind, atmospheric perspective",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Northern lights dancing over a snow-covered forest, starry night sky, lone wolf howling, moonlight casting blue shadows, digital art",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Underwater city with art deco architecture, schools of iridescent fish swimming between buildings, caustic lighting, octane render",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Desert oasis with ancient ruins, palm trees swaying in the wind, golden hour lighting, sand particles in the air, cinematic composition",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Enchanted greenhouse filled with glowing flowers, butterflies with luminescent wings, dew drops on leaves, magical realism style",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    ),
    PromptConfig(
        prompt="Floating islands in the sky connected by crystalline bridges, waterfalls cascading into the void, ethereal clouds, fantasy concept art",
        width=1024,
        height=1024,
        num_inference_steps=4,
        guidance_scale=0,
        seed=None
    )
]
