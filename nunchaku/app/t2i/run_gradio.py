# Changed from https://huggingface.co/spaces/playgroundai/playground-v2.5/blob/main/app.py
import argparse
import os
import random
import time

import GPUtil
import gradio as gr
import spaces
import torch
from peft.tuners import lora

from nunchaku.models.safety_checker import SafetyChecker
from utils import get_pipeline
from vars import DEFAULT_HEIGHT, DEFAULT_WIDTH, EXAMPLES, MAX_SEED, PROMPT_TEMPLATES, SVDQ_LORA_PATHS


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="schnell", choices=["schnell", "dev"], help="Which FLUX.1 model to use"
    )
    parser.add_argument(
        "-p",
        "--precisions",
        type=str,
        default=["int4"],
        nargs="*",
        choices=["int4", "bf16"],
        help="Which precisions to use",
    )
    parser.add_argument("--use-qencoder", action="store_true", help="Whether to use 4-bit text encoder")
    parser.add_argument("--no-safety-checker", action="store_true", help="Disable safety checker")
    parser.add_argument("--count-use", action="store_true", help="Whether to count the number of uses")
    return parser.parse_args()


args = get_args()


pipelines = []
for i, precision in enumerate(args.precisions):
    pipeline = get_pipeline(
        model_name=args.model,
        precision=precision,
        use_qencoder=args.use_qencoder,
        device=f"cuda:{i}",
        lora_name="All",
    )
    pipeline.cur_lora_name = "None"
    pipeline.cur_lora_weight = 0
    pipelines.append(pipeline)

safety_checker = SafetyChecker("cuda", disabled=args.no_safety_checker)


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    lora_name: str = "None",
    lora_weight: float = 1,
    seed: int = 0,
):
    is_unsafe_prompt = False
    if not safety_checker(prompt):
        is_unsafe_prompt = True
        prompt = "A peaceful world."
    prompt = PROMPT_TEMPLATES[lora_name].format(prompt=prompt)
    images, latency_strs = [], []
    for i, pipeline in enumerate(pipelines):
        precision = args.precisions[i]
        progress = gr.Progress(track_tqdm=True)
        if pipeline.cur_lora_name != lora_name:
            if precision == "bf16":
                for m in pipeline.transformer.modules():
                    if isinstance(m, lora.LoraLayer):
                        if pipeline.cur_lora_name != "None":
                            if pipeline.cur_lora_name in m.scaling:
                                m.scaling[pipeline.cur_lora_name] = 0
                        if lora_name != "None":
                            if lora_name in m.scaling:
                                m.scaling[lora_name] = lora_weight
            else:
                assert precision == "int4"
                if lora_name != "None":
                    pipeline.transformer.nunchaku_update_params(SVDQ_LORA_PATHS[lora_name])
                    pipeline.transformer.nunchaku_set_lora_scale(lora_weight)
                else:
                    pipeline.transformer.nunchaku_set_lora_scale(0)
        elif lora_name != "None":
            if precision == "bf16":
                if pipeline.cur_lora_weight != lora_weight:
                    for m in pipeline.transformer.modules():
                        if isinstance(m, lora.LoraLayer):
                            if lora_name in m.scaling:
                                m.scaling[lora_name] = lora_weight
            else:
                assert precision == "int4"
                pipeline.transformer.nunchaku_set_lora_scale(lora_weight)
        pipeline.cur_lora_name = lora_name
        pipeline.cur_lora_weight = lora_weight

        start_time = time.time()
        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).images[0]
        end_time = time.time()
        latency = end_time - start_time
        if latency < 1:
            latency = latency * 1000
            latency_str = f"{latency:.2f}ms"
        else:
            latency_str = f"{latency:.2f}s"
        images.append(image)
        latency_strs.append(latency_str)
    if is_unsafe_prompt:
        for i in range(len(latency_strs)):
            latency_strs[i] += " (Unsafe prompt detected)"
    torch.cuda.empty_cache()

    if args.count_use:
        if os.path.exists("use_count.txt"):
            with open("use_count.txt", "r") as f:
                count = int(f.read())
        else:
            count = 0
        count += 1
        print(f"Use count: {count}")
        with open("use_count.txt", "w") as f:
            f.write(str(count))

    return *images, *latency_strs


with open("./assets/description.html", "r") as f:
    DESCRIPTION = f.read()
gpus = GPUtil.getGPUs()
if len(gpus) > 0:
    gpu = gpus[0]
    memory = gpu.memoryTotal / 1024
    device_info = f"Running on {gpu.name} with {memory:.0f} GiB memory."
else:
    device_info = "Running on CPU 🥶 This demo does not work on CPU."
notice = f'<strong>Notice:</strong>&nbsp;We will replace unsafe prompts with a default prompt: "A peaceful world."'

with gr.Blocks(
    css_paths=[f"assets/frame{len(args.precisions)}.css", "assets/common.css"],
    title=f"SVDQuant FLUX.1-{args.model} Demo",
) as demo:
    gr.HTML(DESCRIPTION.format(model=args.model, device_info=device_info, notice=notice))
    with gr.Row():
        image_results, latency_results = [], []
        for i, precision in enumerate(args.precisions):
            with gr.Column():
                gr.Markdown(f"# {precision.upper()}", elem_id="image_header")
                with gr.Group():
                    image_result = gr.Image(
                        format="png",
                        image_mode="RGB",
                        label="Result",
                        show_label=False,
                        show_download_button=True,
                        interactive=False,
                    )
                    latency_result = gr.Text(label="Inference Latency", show_label=True)
                    image_results.append(image_result)
                    latency_results.append(latency_result)
    with gr.Row():
        prompt = gr.Text(
            label="Prompt", show_label=False, max_lines=1, placeholder="Enter your prompt", container=False, scale=4
        )
        run_button = gr.Button("Run", scale=1)
    if args.model == "dev":
        with gr.Row():
            lora_name = gr.Dropdown(label="LoRA Name", choices=PROMPT_TEMPLATES.keys(), value="None", scale=1)
            prompt_template = gr.Textbox(
                label="LoRA Prompt Template", value=PROMPT_TEMPLATES["None"], scale=1, max_lines=1
            )
    else:
        lora_name = "None"

    with gr.Row():
        seed = gr.Slider(label="Seed", show_label=True, minimum=0, maximum=MAX_SEED, value=233, step=1, scale=4)
        randomize_seed = gr.Button("Random Seed", scale=1, min_width=50, elem_id="random_seed")
    with gr.Accordion("Advanced options", open=False):
        with gr.Group():
            if args.model == "schnell":
                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=8, step=1, value=4)
                guidance_scale = 0
                lora_weight = 0
            elif args.model == "dev":
                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=50, step=1, value=25)
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, step=0.1, value=3.5)
                lora_weight = gr.Slider(label="LoRA Weight", minimum=0, maximum=2, step=0.1, value=1)
            else:
                raise NotImplementedError(f"Model {args.model} not implemented")
    if args.model == "schnell":

        def generate_func(prompt, num_inference_steps, seed):
            return generate(
                prompt, DEFAULT_HEIGHT, DEFAULT_WIDTH, num_inference_steps, guidance_scale, lora_name, lora_weight, seed
            )

        input_args = [prompt, num_inference_steps, seed]
    elif args.model == "dev":

        def generate_func(prompt, num_inference_steps, guidance_scale, lora_name, lora_weight, seed):
            return generate(
                prompt, DEFAULT_HEIGHT, DEFAULT_WIDTH, num_inference_steps, guidance_scale, lora_name, lora_weight, seed
            )

        input_args = [prompt, num_inference_steps, guidance_scale, lora_name, lora_weight, seed]

    gr.Examples(
        examples=EXAMPLES[args.model], inputs=input_args, outputs=[*image_results, *latency_results], fn=generate_func
    )

    gr.on(
        triggers=[prompt.submit, run_button.click],
        fn=generate_func,
        inputs=input_args,
        outputs=[*image_results, *latency_results],
        api_name="run",
    )
    randomize_seed.click(
        lambda: random.randint(0, MAX_SEED), inputs=[], outputs=seed, api_name=False, queue=False
    ).then(fn=generate_func, inputs=input_args, outputs=[*image_results, *latency_results], api_name=False, queue=False)

    if args.model == "dev":
        lora_name.change(
            lambda x: PROMPT_TEMPLATES[x],
            inputs=[lora_name],
            outputs=[prompt_template],
            api_name=False,
            queue=False,
        ).then(
            fn=generate_func, inputs=input_args, outputs=[*image_results, *latency_results], api_name=False, queue=False
        )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0", debug=True, share=True)
