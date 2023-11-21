from PIL import Image
import numpy as np
import torch
from torchvision.transforms import RandomErasing

img = Image.open('img.JPG')
print(type(img))  
print(np.array(img).shape)）

# img.show()

from torchvision.transforms import Resize

resize = Resize(
    size=(224, 224),  # (height, width)
    interpolation=2
)
# resize(img).show()


from torchvision.transforms import CenterCrop,RandomCrop
from PIL import Image

center_crop = CenterCrop(size=(224, 224))
# center_crop(img).show()

random_crop = RandomCrop(
    size=(224, 224),  
    padding=(75, 75), 
    pad_if_needed=False,  
    fill=(255, 0, 0),  
    padding_mode='constant' 
)
# random_crop(img).show()


from torchvision.transforms import RandomRotation

random_rotation = RandomRotation(
    degrees=(-45, 45), 
    resample=False,  
    expand=True,  
    center=(0, 0)  
)

from torchvision.transforms import RandomHorizontalFlip
image = Image.open('img.JPG')

random_horizontal_flip = RandomHorizontalFlip(p=1.0)  
# random_horizontal_flip(image).show()  

from torchvision.transforms import RandomVerticalFlip

random_vertical_flip = RandomVerticalFlip(p=1.0)
# random_vertical_flip(image).show() 

from PIL import Image
from torchvision.transforms import Grayscale,ColorJitter

grayscale = Grayscale(
    num_output_channels=3  # num_output_channels should be either 1 or 3
)
# grayscale(image).show() 

color_jitter = ColorJitter(
    brightness=(1.5, 1.5), 
                            
    contrast=0, 
    saturation=0,  
    hue=0  

)
# color_jitter(image).show() 

color_jitter = ColorJitter(
    brightness=0,
    contrast=0,
    saturation=(1.5, 1.5),
    hue=0
)
# color_jitter(image).show() 


import random
from PIL import Image
import skimage.io
import skimage.util
import numpy as np


class RandomNoise:
 
    def __init__(self, modes, p=0.5):

        self.modes = modes
        self.p = p

    def __call__(self, image):
        if random.uniform(0, 1) < self.p:  
            img_arr = np.array(image)
            for mode in self.modes:
                img_arr = skimage.util.random_noise(img_arr, mode)

            img_pil = Image.fromarray((img_arr * 255).astype(np.uint8))

            return img_pil
        else:
            return image


modes = ['gaussian'] *10
# modes = ['gaussian', 'pepper', 'speckle']
random_noise = RandomNoise(modes, p=1.)
noisy_image = random_noise(image)
noisy_image.show()


# modes = ['gaussian', 'pepper', 'speckle'] * 10


