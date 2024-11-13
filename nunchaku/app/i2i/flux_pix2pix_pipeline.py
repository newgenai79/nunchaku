import os
from typing import Any, Callable, Optional, Union

import torch
import torchvision.transforms.functional as F
import torchvision.utils
from PIL import Image
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, FluxPipelineOutput, FluxTransformer2DModel
from einops import rearrange
from huggingface_hub import hf_hub_download
from peft.tuners import lora
from torch import nn

from nunchaku.models.flux import inject_pipeline, load_quantized_model
from nunchaku.pipelines.flux import quantize_t5


class FluxPix2pixTurboPipeline(FluxPipeline):
    def update_alpha(self, alpha: float) -> None:
        self._alpha = alpha
        transformer = self.transformer

        for n, p in transformer.named_parameters():
            if n in self._tuned_state_dict:
                new_data = self._tuned_state_dict[n] * alpha + self._original_state_dict[n] * (1 - alpha)
                new_data = new_data.to(self._execution_device).to(p.dtype)
                p.data.copy_(new_data)
        if self.precision == "bf16":
            for m in transformer.modules():
                if isinstance(m, lora.LoraLayer):
                    m.scaling["default_0"] = alpha
        else:
            assert self.precision == "int4"
            transformer.nunchaku_set_lora_scale(alpha)

    def load_control_module(
        self,
        pretrained_model_name_or_path: str,
        weight_name: str | None = None,
        svdq_lora_path: str | None = None,
        alpha: float = 1,
    ):
        state_dict, alphas = self.lora_state_dict(
            pretrained_model_name_or_path, weight_name=weight_name, return_alphas=True
        )

        transformer = self.transformer
        original_state_dict = {}
        tuned_state_dict = {}
        assert isinstance(transformer, FluxTransformer2DModel)

        for n, p in transformer.named_parameters():
            if f"transformer.{n}" in state_dict:
                original_state_dict[n] = p.data.cpu()
                tuned_state_dict[n] = state_dict[f"transformer.{n}"].cpu()

        self._original_state_dict = original_state_dict
        self._tuned_state_dict = tuned_state_dict
        if self.precision == "bf16":
            self.load_lora_into_transformer(state_dict, {}, transformer=transformer)
        else:
            assert svdq_lora_path is not None
            self.transformer.nunchaku_update_params(svdq_lora_path)
        self.update_alpha(alpha)

    @torch.no_grad()
    def __call__(
        self,
        image: str or Image,
        image_type: str = "sketch",
        alpha: float = 1.0,
        prompt: str | None = None,
        prompt_2: str | None = None,
        height: int | None = 1024,
        width: int | None = 1024,
        timesteps: list[int] = None,
        generator: torch.Generator | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if alpha != self._alpha:
            self.update_alpha(alpha)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        guidance_scale = 0

        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((width, height), Image.LANCZOS)
        else:
            image = image.resize((width, height), Image.LANCZOS)
        image_t = F.to_tensor(image) < 0.5
        image_t = image_t.unsqueeze(0).to(self.dtype).to(device)

        kernel_size = 4
        if hasattr(self, "erosion_kernel"):
            erosion_kernel = self.erosion_kernel
        else:
            erosion_kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=self.dtype, device=device)
            self.erosion_kernel = erosion_kernel

        torchvision.utils.save_image(image_t[0], "before.png")
        image_t = (
            nn.functional.conv2d(image_t[:, :1], erosion_kernel, padding=kernel_size // 2) > kernel_size**2 - 0.1
        )
        image_t = torch.concat([image_t, image_t, image_t], dim=1).to(self.dtype)
        torchvision.utils.save_image(image_t[0], "after.png")

        image_t = (image_t - 0.5) * 2

        # 4. Prepare latent variables
        encoded_image = self.vae.encode(image_t, return_dict=False)[0].sample(generator=generator)
        encoded_image = (encoded_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        if generator is None:
            z = torch.randn_like(encoded_image)
        else:
            z = torch.randn(
                encoded_image.shape, device=generator.device, dtype=encoded_image.dtype, generator=generator
            ).to(device)
        noisy_latent = z * (1 - alpha) + encoded_image * alpha
        noisy_latent = rearrange(noisy_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        num_channels_latents = self.transformer.config.in_channels // 4
        _, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )

        # 5. Denoising
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = torch.full((batch_size,), 1.0, dtype=self.dtype, device=device)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(noisy_latent.shape[0])
        else:
            guidance = None

        pred = self.transformer(
            hidden_states=noisy_latent,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        encoded_output = noisy_latent - pred

        if output_type == "latent":
            image = encoded_output

        else:
            encoded_output = self._unpack_latents(encoded_output, height, width, self.vae_scale_factor)
            encoded_output = (encoded_output / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(encoded_output, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        qmodel_device = kwargs.pop("qmodel_device", "cuda:0")
        qmodel_device = torch.device(qmodel_device)

        qmodel_path = kwargs.pop("qmodel_path", None)
        qencoder_path = kwargs.pop("qencoder_path", None)

        pipeline = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipeline.precision = "bf16"

        if qmodel_path is not None:
            assert isinstance(qmodel_path, str)
            if not os.path.exists(qmodel_path):
                hf_repo_id = os.path.dirname(qmodel_path)
                filename = os.path.basename(qmodel_path)
                qmodel_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
            m = load_quantized_model(qmodel_path, 0 if qmodel_device.index is None else qmodel_device.index)
            inject_pipeline(pipeline, m)
            pipeline.precision = "int4"

        if qencoder_path is not None:
            assert isinstance(qencoder_path, str)
            if not os.path.exists(qencoder_path):
                hf_repo_id = os.path.dirname(qencoder_path)
                filename = os.path.basename(qencoder_path)
                qencoder_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
            quantize_t5(pipeline, qencoder_path)

        return pipeline
