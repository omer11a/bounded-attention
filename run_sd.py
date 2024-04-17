import os
import torch
import torchvision.transforms.functional as F

from diffusers import DDIMScheduler
from pipeline_stable_diffusion_opt import StableDiffusionPipeline
from pytorch_lightning import seed_everything

from injection_utils import register_attention_editor_diffusers
from bounded_attention import BoundedAttention
import utils


def load_model(device):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}, torch_dtype=torch.float16, use_safetensors=True).to(device)
    model.enable_xformers_memory_efficient_attention()
    model.enable_sequential_cpu_offload()
    return model


def run(
    boxes,
    prompt,
    subject_token_indices,
    out_dir='out',
    seed=286,
    batch_size=1,
    filter_token_indices=None,
    eos_token_index=None,
    init_step_size=30,
    final_step_size=10,
    first_refinement_step=15,
    num_clusters_per_subject=3,
    cross_loss_scale=1,
    self_loss_scale=1,
    classifier_free_guidance_scale=7.5,
    num_gd_iterations=5,
    loss_threshold=0.2,
    num_guidance_steps=15,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device)

    seed_everything(seed)
    prompts = [prompt] * batch_size
    start_code = torch.randn([len(prompts), 4, 64, 64], device=device)

    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    out_dir = os.path.join(out_dir, f"sample_{sample_count}")
    os.makedirs(out_dir)

    editor = BoundedAttention(
        boxes,
        prompts,
        subject_token_indices,
        list(range(12, 20)),
        list(range(12, 20)),
        cross_mask_layers=list(range(14, 20)),
        self_mask_layers=list(range(14, 20)),
        filter_token_indices=filter_token_indices,
        eos_token_index=eos_token_index,
        cross_loss_coef=cross_loss_scale,
        self_loss_coef=self_loss_scale,
        max_guidance_iter=num_guidance_steps,
        max_guidance_iter_per_step=num_gd_iterations,
        start_step_size=init_step_size,
        end_step_size=final_step_size,
        loss_stopping_value=loss_threshold,
        min_clustering_step=first_refinement_step,
        num_clusters_per_box=num_clusters_per_subject,
    )

    register_attention_editor_diffusers(model, editor)
    images = model(prompts, latents=start_code, guidance_scale=classifier_free_guidance_scale)

    for i, image in enumerate(images):
        image = F.to_pil_image(image)
        image.save(os.path.join(out_dir, f'{seed}_{i}.png'))
        utils.draw_box(image, boxes).save(os.path.join(out_dir, f'{seed}_{i}_boxes.png'))

    print("Syntheiszed images are saved in", out_dir)


def main():
    boxes = [
        [0.05, 0.2, 0.45, 0.8],
        [0.55, 0.2, 0.95, 0.8],
    ]

    prompt = "A ginger kitten and a gray puppy in a yard"
    subject_token_indices = [[2, 3], [6, 7]]

    run(boxes, prompt, subject_token_indices, init_step_size=8, final_step_size=2)


if __name__ == "__main__":
    main()
