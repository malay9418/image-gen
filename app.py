# import gradio as gr
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch


prj_path = "malay-91418/profile-pic-gen"
model = "stabilityai/stable-diffusion-xl-base-1.0"
seed = 42

pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    cache_dir="./models"
)
pipe.to("cuda")
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to("cuda")
generator = torch.Generator("cuda").manual_seed(seed)


def generate_image(prompt):
    image = pipe(prompt=prompt, generator=generator).images[0]
    image = refiner(prompt=prompt, generator=generator, image=image).images[0]
    return image


# iface = gr.Interface(
#    fn=generate_image,
#    inputs=[gr.Textbox(label="Prompt")],
#    outputs=gr.Image(label="Generated Image"),
# )

# iface.launch(share=True)
