# Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation

> **Omer Dahary, Or Patashnik, Kfir Aberman, Daniel Cohen-Or**
> 
> Text-to-image diffusion models have an unprecedented ability to generate diverse and high-quality images. However, they often struggle to faithfully capture the intended semantics of complex input prompts that include multiple subjects. Recently, numerous layout-to-image extensions have been introduced to improve user control, aiming to localize subjects represented by specific tokens. Yet, these methods often produce semantically inaccurate images, especially when dealing with multiple semantically or visually similar subjects. In this work, we study and analyze the causes of these limitations. Our exploration reveals that the primary issue stems from inadvertent semantic leakage between subjects in the denoising process. This leakage is attributed to the diffusion model’s attention layers, which tend to blend the visual features of different subjects. To address these issues, we introduce Bounded Attention, a training-free method for bounding the information flow in the sampling process. Bounded Attention prevents detrimental leakage among subjects and enables guiding the generation to promote each subject's individuality, even with complex multi-subject conditioning. Through extensive experimentation, we demonstrate that our method empowers the generation of multiple subjects that better align with given prompts and layouts.

<a href="https://omer11a.github.io/bounded-attention/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2403.16990"><img src="https://img.shields.io/badge/arXiv-LPM-b31b1b.svg" height=20.5></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/omer11a/bounded-attention)

<p align="center">
<img src="images/teaser.jpg" width="800px"/>
</p>

## Description  
Official implementation of our "Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation" paper.

## Setup

### Dependencies
To install dependencies, please run:

```
pip install -r requirements.txt
```

### Demo

This project has a gradio [demo](https://huggingface.co/spaces/omer11a/bounded-attention) deployed in HuggingFace.
To run the demo locally, run the following: 
```shell
gradio app.py
```
Then, you can connect to the local demo by browsing to `http://localhost:7860/`.

## Usage

### Basics

<p align="center">
<img src="images/example.jpg" width="800px"/>  
<br>
Example generations by SDXL with and without Bounded Attention.
</p>


To generate images, you can run `run_xl.py` for our SDXL version, and `run_sd.py` for our Stable Diffusion version.
In each script, we call the `run` function to generate the images. E.g.,
```
boxes = [
    [0.35, 0.4, 0.65, 0.9],
    [0, 0.6, 0.3, 0.9],
    [0.7, 0.55, 1, 0.85],
]

prompt = "3 D Pixar animation of a cute unicorn and a pink hedgehog and a nerdy owl traveling in a magical forest"
subject_token_indices = [[7, 8, 17], [11, 12, 17], [15, 16, 17]]

run(boxes, prompt, subject_token_indices, init_step_size=25, final_step_size=10)
```

The `run` function recieves the following parameters:
- boxes: the bounding box of each subject in the format [(x0, y0, x1, x2), ...], where the top-left corner is x=0,y=0 and the bottom-right corner is x=1,y=1.
- prompt: the textual prompt.
- subject_token_indices: The indices of each token belonging to each subject, where the indices start from 1. Tokens can be shared between subjects.
- out_dir: The output directory. Defaults to "out".
- seed: The random seed.
- batch_size: The number of generated images.
- filter_token_indices: The indices of the tokens to ignore. This is automatically inferred, but we recommend explicitly ignoring prepositions, numbers and positional relations.
- eos_token_index: The index of the EOS token (the first padding token appended to the end of the prompt). This is automatically inferred, but we recommend explicitly passing it, as we use it to verify you have correctly counted the number of tokens.

### Advanced options

The `run` function also supports the following optional hyperparameters:

- init_step_size: The initial step size of the linear step size scheduler when performing guidance.
- final_step_size: The final step size of the linear step size scheduler when performing guidance.
- num_clusters_per_subject: The number of clusters computed when clustering the self-attention maps (#clusters = #subject x #clusters_per_subject). Changing this value might improve semantics (adherence to the prompt), especially when the subjects exceed their bounding boxes.
- cross_loss_scale: The scale factor of the cross-attention loss term. Increasing it will improve semantic control (adherence to the prompt), but may reduce image quality.
- self_loss_scale: The scale factor of the self-attention loss term. Increasing it will improve layout control (adherence to the bounding boxes), but may reduce image quality.
- classifier_free_guidance_scale: The scale factor of classifier-free guidance.
- num_guidance_steps: The number of timesteps in which to perform guidance. Decreasing this also decreases the runtime.
- first_refinement_step: The timestep from which subject mask refinement is performed.
- num_gd_iterations: The number of Gradient Descent iterations for each timestep when performing guidance.
- loss_threshold: If the loss is below the threshold, Gradient Descent stops for that timestep.

## Acknowledgements 

This code was built using the code from the following repositories:
- [diffusers](https://github.com/huggingface/diffusers)
- [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/)
- [MasaCtrl](https://github.com/TencentARC/MasaCtrl)

## Citation

If you use this code for your research, please cite our paper:

```
@misc{dahary2024yourself,
    title={Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation},
    author={Omer Dahary and Or Patashnik and Kfir Aberman and Daniel Cohen-Or},
    year={2024},
    eprint={2403.16990},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
 }
```
