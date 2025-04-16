from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-ema-celebahq-256"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to("cuda")

print(help(ddpm.unet))

# # run pipeline in inference (sample random noise and denoise)
# image = ddpm().images
#
#
#
#
# # save image
# image[0].save("ddpm_generated_image.png")