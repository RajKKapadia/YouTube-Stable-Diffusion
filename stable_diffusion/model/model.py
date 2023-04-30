import numpy as np
from diffusers import StableDiffusionPipeline
from torch import autocast
import base64
from io import BytesIO

import torch
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipeline = StableDiffusionPipeline.from_pretrained(
    'model/',
    torch_dtype=torch.float16
)

pipe = pipeline.to(device)


def generate_image_from_text(prompt: str, steps: int = 10) -> list:
    '''
    Generate an anime images using Stable Deffusion from the prompt. This will take a piece of text as an input and returns list of object.

    Parameters:
        - prompt(str): a piece of text used to generate an image

    Returrns:
        - list: Returns a list of object, each object has two properties, one unsafe_prompt, either true or false, and two image, a PIL Image object.
    '''
    SEED = np.random.randint(2023)
    generator = torch.Generator(device).manual_seed(SEED)
    pipe.safety_checker = None
    with autocast(device):
        result = pipe(
            prompt,
            num_inference_steps=steps,
            generator=generator,
            height=128,
            width=128
        )

    images = result.images
    output = []

    if len(images) == 0:
        output.append(
            {
                'status': 0,
                'image': []
            }
        )
    else:
        output.append(
            {
                'status': 1,
                'image': images[0]
            }
        )

    return output


def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format='JPEG')
    img_str = base64.b64encode(buff.getvalue())
    return img_str


def generate_image(data: dict) -> dict:
    '''
    Mapper function to generate images
    '''
    if 'steps' in data.keys():
        result = generate_image_from_text(data['prompt'], data['steps'])
    else:
        result = generate_image_from_text(data['prompt'])
    if result[0]['status'] == 1:
        image = result[0]['image']
        base = im_2_b64(image)
        return {
            'status': 1,
            'message': 'Request successful',
            'image': f'data:image/jpeg;base64,{base.decode()}'
        }
    else:
        return {
            'status': 0,
            'message': 'Error at model level.',
            'image': 'None'
        }
