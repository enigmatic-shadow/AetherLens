import os
import time
import functools
from typing import Any
import io
import warnings
from collections import namedtuple

import segment_anything
import gradio as gr
import numpy as np
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
)
import torch
import qrcode
import openai
from diffusers import StableDiffusionInpaintPipeline




controlnet = ControlNetModel.from_pretrained(
    "monster-labs/control_v1p_sd15_qrcode_monster",
    torch_dtype=torch.float16

)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16


)
# pipe = pipe.to('cuda')
# controlnet = controlnet.to('cuda')

inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    # revision="fp16",
    torch_dtype=torch.float16,
)
# inpainting_pipe.to("cuda")


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}



# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.

    engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation.
    # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

openai.api_key = os.environ['OPENAI_API_KEY']



sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"


sam = segment_anything.sam_model_registry[model_type](checkpoint='sam_vit_h_4b8939.pth')
predictor = segment_anything.SamPredictor(sam)
sam.to(device=device)

def timer(func:callable) -> callable:
  @functools.wraps(func)
  def wrapper(*args, **kwargs) -> Any:
    t1 = time.perf_counter()
    res = func(*args,**kwargs)
    t2 = time.perf_counter() - t1
    print(f'{func.__name__}: {t2:.2f} sec(s)')
    return res
  return wrapper


@timer
def img_without_background(prompt, img):
  rgba = Image.fromarray(img)
  rgba.save('rgba.png')
  rgba = Image.open('rgba.png')
  
  
  
  mask_image = Image.fromarray(np.array(rgba)[:, :, 3] == 0)
  
  
  # run the pipeline
  
  # image and mask_image should be PIL images.
  # The mask structure is white for outpainting and black for keeping as is
  image = pipe(
      prompt=prompt,
      image=rgba,
      mask_image=mask_image,
      # guidance_scale = 5.0
  ).images[0]
  return image

@timer
def gpt(user_query:str)->str:
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"""Generate most suitable for a vehicle depending on the description:

                car_description: a photo of a vibrant car in desert
                tagline: Unleash Your Adventurous Spirit: Conquer the Desert in Vibrant Style

                car_description: a vibrant green sedan in a rain forest
                tagline: Embrace Nature's Beauty: Journey Through the Rainforest in Vibrant Green Elegance

                car_description: {user_query}
                tagline:""",
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
              )
  return response['choices'][0]['text']

def create_code(content: str):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=16,
        border=0,
    )
    qr.add_data(content)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # find smallest image size multiple of 256 that can fit qr
    offset_min = 8 * 16
    w, h = img.size
    w = (w + 255 + offset_min) // 256 * 256
    h = (h + 255 + offset_min) // 256 * 256
    if w > 1024:
        raise gr.Error("QR code is too large, please use a shorter content")
    bg = Image.new('L', (w, h), 128)

    # align on 16px grid
    coords = ((w - img.size[0]) // 2 // 16 * 16,
              (h - img.size[1]) // 2 // 16 * 16)
    bg.paste(img, coords)
    return bg


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    seed: int = -1,
    sampler="Euler a",
):


    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    print("Generating QR Code from content")
    qrcode_image = create_code(qr_code_content)

    # hack due to gradio examples
    init_image = qrcode_image

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        width=qrcode_image.width,
        height=qrcode_image.height,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),

        num_inference_steps=40,
    )
    return out.images[0]


@timer
def add_white_background(img:np.array,evt:gr.SelectData):

  # Load the image with transparency (alpha)
  img = Image.fromarray(img)
  # Create a new white background image with the same size as the original image
  width, height = img.size
  white_background = Image.new("RGBA", img.size, (255, 255, 255, 255))

  # If the image has an alpha channel, paste the RGB channels and use the alpha channel as the mask
  if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
      white_background.paste(img, (0, 0), img.split()[3])
  else:
      white_background.paste(img, (0, 0))

  return white_background


@timer
def _check_dimension(img:np.ndarray) -> Image.Image:
  def nearest_mutltiple_of_64(n:int) -> int:
    return n if (n%64 == 0)  else ((n // 64) * 64)
  img = Image.fromarray(img)
  og_dimensions = img.size
  print(f'{og_dimensions = }')
  updated_dimensions = list(map(nearest_mutltiple_of_64, og_dimensions))
  print(f'{updated_dimensions = }')
  return img.resize(updated_dimensions)#, og_dimensions, (og_dimensions == updated_dimensions)

## @timer
# def _check_dimension(img:np.ndarray) -> Image.Image:
#   img = Image.fromarray(img)
#   dimensions = img.size
#   print(dimensions)
#   is_multiples_of_64 = (filter)
#   return (img, dimensions, False) if is_multiples_of_64 else (img.resize((1024,1024)), dimensions, True)


@timer
def inpainting(prompt: str, init_image: np.ndarray, mask_image: np.array) -> Image.Image:

  # width, height = init_image.shape
  # print(f"Init Image: {width = }, {height = }")
  mask_image[mask_image == 255] = 125
  mask_image[mask_image == 0] = 255
  mask_image[mask_image == 125] = 0
  print(init_image.__class__)

  # ImageClass = namedtuple("ImageClass",["img", "og_dimensions", "dimesions_changed"])

  # init_image =  ImageClass(*_check_dimension(init_image))
  mask_image = Image.fromarray(mask_image)#.resize(init_image.img.size)
  print(f'mask_image.dimensions = {mask_image.size}')
  mask_image.save('mask_inpaint.png')
#   grpc._channel._MultiThreadedRendezvous: <_MultiThreadedRendezvous of RPC that terminated with:
# 	status = StatusCode.UNKNOWN
# 	details = "mask must have the same dimensions as the image"
# 	debug_error_string = "UNKNOWN:Error received from peer ipv4:104.18.20.212:443 {created_time:"2023-07-28T10:44:21.161539605+00:00", grpc_status:2, grpc_message:"mask must have the same dimensions as the image"}"
# >
  print(f"mask_image: {mask_image.__class__}")
  answers = stability_api.generate(
    prompt = prompt,
    init_image = Image.fromarray(init_image),
    mask_image = mask_image ,
    start_schedule=1,
    seed=44332211, # If attempting to transform an image that was previously generated with our API,
                   # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
    steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=1024, # Generation width, if not included defaults to 512 or 1024 depending on the engine.
    height=1024, # Generation height, if not included defaults to 512 or 1024 depending on the engine.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )


  # Set up our warning to print to the console if the adult content classifier is tripped.
  # If adult content classifier is not tripped, save generated image.
  for resp in answers:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn(
                  "Your request activated the API's safety filters and could not be processed."
                  "Please modify the prompt and try again.")
          if artifact.type == generation.ARTIFACT_IMAGE:

              img = Image.open(io.BytesIO(artifact.binary))

              img.save(f"{artifact.seed}.png") # Save our completed image with its seed number as the filename.
  display(img)
  # print(f'Dimensions have changed: {init_image.og_dimensions}')
  # print(f'returned image dimensions: {img.size}')

  # img.resize(init_image.og_dimensions)

  return img







selected_pixels = []

@timer
def generate_mask(image:np.array,evt:gr.SelectData)->Image.Image:
  print('-'*50)
  # t1 = time.perf_counter()
  print(f"image inpainting: {image}")
  print('ENter fucntion')

  selected_pixels.append(evt.index)
  print(f'Selected {selected_pixels}')

  predictor.set_image(image)

  print('After predictor.set_image(')
  input_points = np.array(selected_pixels)
  input_labels = np.ones(input_points.shape[0])
  print('After labels')
  mask, _, _ = predictor.predict(
      point_coords = input_points,
      point_labels = input_labels,
      multimask_output = False
  )
  print(f'mask: {mask}')
  mask = np.logical_not(mask)
  mask = Image.fromarray(mask[0,:,:])
  print(mask.height, mask.width)
  print(f'RETURNING TYPE{mask.__class__}')
  mask.save(f"mask.png")
  # print(f'{time.perf_counter()-t1} seconds')

  return mask

def send_image(prompt, input_image, mask_image, has_bg ):
  if has_bg: 
    print('img_without_background')
    img = img_without_background(prompt,input_image)
  else:
    print('mask_image')
    img = inpainting(prompt,init_image, mask_image)
  return img

theme = gr.themes.Glass(
    primary_hue="cyan",
    secondary_hue="violet",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme,css=".gradio-container {background-color: #0070ad } ") as demo:
  gr.Markdown("""<h1 style="font-family:Copperplate;text-align:center;color: white">AetherLens</h1>""")
  gr.Markdown("""<h4 style="font-family:Copperplate;text-align:center;color: white">Unveil the Ethereal: Elevate Your Vehicle's Aura</h4>""")  
  gr.HTML("""<p style='font-family:Garamond;font-size: 20px; color: white'>Are you tired of using the same old, generic images for marketing your vehicles on social media? Do you want your vehicles to truly stand out and make a lasting impression on potential customers? Look no further! AetherLens is here to take your marketing and social media game to the next level.
AetherLens is not just another photo editing app; it's a state-of-the-art, cutting-edge application that is specifically tailored for showcasing your vehicles in a whole new light. Our team of expert developers and designers have meticulously crafted AetherLens to revolutionize the way you present your vehicles to the world.
With AetherLens, the possibilities are endless. Say goodbye to mundane and boring vehicle pictures. Now, you can effortlessly create stunning images with captivating backgrounds that perfectly complement your vehicles. Whether it's a sleek city backdrop that adds a touch of sophistication, a breathtaking mountain landscape that instills a sense of adventure, or a dreamy beach scene that sparks wanderlust - AetherLens has it all.
Our powerful AI-based technology makes changing backgrounds a breeze. No need to spend hours trying to get the perfect shot. With a few taps on your device, you can magically transform your vehicle images into attention-grabbing masterpieces that are bound to turn heads on social media.</p>""")
  gr.HTML(value="<img id='HeadImage' src='https://i.ibb.co/fHC4H2H/image.png' width='1200' height='300' style='border: 2px solid ##2B0A3D;'/>")
  gr.HTML(value="<style>#HeadImage:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")
  gr.HTML(value="<style>#ImageAcc1{border: 2px solid ##2B0A3D;}</style>")
  with gr.Tab('Customize your car!'):
    with gr.Accordion(""):
      gr.HTML("""<p style='font-family:Garamond;font-size: 20px; color: black'>Kindly ensure that the images you upload have dimensions that are multiples of 64. This is important because the stability API can only handle images with such dimensions effectively.</p>""")
      with gr.Row():
        with gr.Column():
          prompt = gr.Textbox(label = "Prompt: ")
          has_bg = gr.Radio(choices=["Is image without backgound"], label=" ",info="This feature needs GPU")
          input_image = gr.Image(label="Edit Car",type="numpy")
        with gr.Column():
          mask_image = gr.Image(label="Generated Mask")#type = "pil")
          output_image = gr.Image(label="Car",type="pil")
      with gr.Row():
        with gr.Column():
          change = gr.Button("Generate")
        with gr.Column():
          clr_btn = gr.Button("Clear Mask")

    with gr.Accordion('Generate Taglines'):
      with gr.Row():
        with gr.Column():
          car_descrip = gr.Textbox(label = 'Description')
        with gr.Column():
          tagline = gr.Textbox(label="Your Awesome Tagline", interactive = False)
      generate_tagline_btn = gr.Button("Generate Tagline")

    generate_tagline_btn.click(gpt, car_descrip, tagline)

    # input_image.upload(_check_dimension, input_image, input_image )
    input_image.select(generate_mask, input_image,mask_image)
    # has_bg.select( add_white_background, input_image, input_image )
    clr_btn.click(
        lambda: selected_pixels.clear(), None,None
    )
    change.click(
        send_image, [prompt, input_image, mask_image,has_bg],
        [output_image]
    )
    
    gr.Examples(
        examples=[
            [
                "rain forest",
                "examples/Porsche911-removebg-preview.png",
                
                
                "Is image without backgound",
                "examples/porsche-forest.png"
                
            ],
            [
                "snowy road",
                "examples/og-jeep.png",
                
                
                "",
                "examples/changed-jeep.png"
                
            ],
            [
                "rain forest",
                "examples/race-car.png",
                "",
                "examples/image (11).png"

            ],
            [
                "new york city",
                "examples/og-red-suv.png",
                "",
                "examples/changed-red-suv.png"
            ]

        ],
        fn = send_image,
        inputs = [prompt,input_image,has_bg,output_image],
        outputs = [output_image]

    )
  with gr.Tab('QR Code'):
    gr.HTML("""<p style='font-family:Garamond;font-size: 20px; color: black'>This requires GPU.</p>""")  
    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="QR Code Content or URL",
                info="The text you want to encode into the QR code",
                value="",
            )

            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="",#ugly, disfigured, low quality, blurry, nsfw",
                info="Prompt that guides the generation away from",
            )

            with gr.Group(
                # label="Params: The generated QR Code functionality is largely influenced by the parameters detailed below",
                # open=True,
            ):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.5,
                    maximum=2.5,
                    step=0.01,
                    value=1.5,
                    label="Controlnet Conditioning Scale",
                    info="""Controls the readability/creativity of the QR code.
                    High values: The generated QR code will be more readable.
                    Low values: The generated QR code will be more creative.
                    """
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=25.0,
                    step=0.25,
                    value=7,
                    label="Guidance Scale",
                    info="Controls the amount of guidance the text prompt guides the image generation"
                )
                sampler = gr.Dropdown(choices=list(
                    SAMPLER_MAP.keys()), value="Euler a", label="Sampler")
                seed = gr.Number(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                    info="Seed for the random number generator. Set to -1 for a random seed"
                )
            
                
        with gr.Column():
            result_image = gr.Image(label="QR Code", elem_id="result_image")
    run_btn = gr.Button("Generate QR Code")
    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            sampler,
        ],
        outputs=[result_image],
    )

    gr.Examples(
        examples=[
            [
                "test",
                "Baroque rococo architecture, architectural photography, post apocalyptic New York, hyperrealism, [roots], hyperrealistic, octane render, cinematic, hyper detailed, 8K",
                "",
                7,
                1.6,
                2592353769,
                "Euler a",
            ],
            [
                "https://qrcodemonster.art",
                "a centered render of an ancient tree covered in bio - organic micro organisms growing in a mystical setting, cinematic, beautifully lit, by tomasz alen kopera and peter mohrbacher and craig mullins, 3d, trending on artstation, octane render, 8k",
                "",
                7,
                1.57,
                259235398,
                "Euler a",
            ],
            [
                "test",
                "3 cups of coffee with coffee beans around",
                "",
                7,
                1.95,
                1889601353,
                "Euler a",
            ],
            [
                "https://huggingface.co",
                "A top view picture of a sandy beach with a sand castle, beautiful lighting, 8k, highly detailed",
                "sky",
                7,
                1.15,
                46200,
                "Euler a",
            ],
            [
                "test",
                "A top view picture of a sandy beach, organic shapes, beautiful lighting, bumps and shadows, 8k, highly detailed",
                "sky, water, squares",
                7,
                1.25,
                46220,
                "Euler a",
            ],
        ],
        fn=inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            seed,
            sampler,
        ],
        outputs=[result_image],

    )





demo.queue(concurrency_count=3,max_size=2)
demo.launch(debug=True)
