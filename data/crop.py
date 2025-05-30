import numpy as np
from PIL import Image
import random
from scipy import stats
from skimage import feature, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import CenterCrop

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    cropped_images = []
    images = []

    x, y = 0, 0 # Initialize x and y
    for y_loop_var in range(0, image.height - window_size + 1, stride):
        for x_loop_var in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x_loop_var, y_loop_var, x_loop_var + window_size, y_loop_var + window_size)))
            x, y = x_loop_var, y_loop_var # Update x and y with the last values from the loop
    
    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray)
        if metric == 'sd':
            m = np.std(crop_gray / 255.0)
        elif metric == 'ghe':
            m = histogram_entropy_response(crop_gray / 255.0)
        elif metric == 'le':
            m = local_entropy_response(crop_gray)
        elif metric == 'ac':
            m = autocorrelation_response(crop_gray / 255.0)
        elif metric == 'td':
            m = texture_diversity_response(crop_gray / 255.0)
        images.append((crop, m))

    images.sort(key=lambda x: x[1], reverse=True)
    
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]

    repeat_images = texture_images.copy()
    while len(texture_images) < n:
        texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])

    return texture_images


def autocorrelation_response(image_array):
    """
    Calculates the average autocorrelation of the input image.
    """
    f = np.fft.fft2(image_array, norm='ortho')
    power_spectrum = np.abs(f) ** 2
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()
    acf = np.mean(acf)

    return acf

def histogram_entropy_response(image):
    """
    Calculates the entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    prob_dist = histogram / histogram.sum()
    entr = stats.entropy(prob_dist + 1e-7, base=2)    # Adding a small value (1e-7) to avoid log(0)

    return entr

def local_entropy_response(image):
    """
    Calculates the spatial entropy of the image using a local entropy filter.
    """
    entropy_image = entropy(image, disk(10))  
    mean_entropy = np.mean(entropy_image)

    return mean_entropy

def texture_diversity_response(image):
    M = image.shape[0]  
    l_div = 0

    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])

    # Vertical differences
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])

    # Diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])

    # Counter-diagonal differences
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])

    return l_div


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    cropped_images = []
    texture_images = []
    images = []

    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))

    if not drop:
        x = x + stride
        y = y + stride

        if x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if x + window_size > image.width and y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray = np.array(crop_gray) / 255.0
        
        histogram, _ = np.histogram(crop_gray.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()
        m = stats.entropy(prob_dist + 1e-7, base=2)
        if m > threshold: 
            texture_images.append(crop)

    if len(texture_images) == 0:
        texture_images = [CenterCrop(image)]

    return texture_images


class crop_base_Rec_Module(nn.Module):
    def __init__(self, window_size=128, stride=16, metric='ghe', drop=False):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.metric = metric
        self.drop = drop
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image_tensor): # image_tensor is C, H, W
        if image_tensor.ndim == 4 and image_tensor.shape[0] == 1: # Handling potential batch dim of 1
            image_tensor = image_tensor.squeeze(0)
        elif image_tensor.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W) or 4D (1, C, H, W), got {image_tensor.shape}")

        image_pil = self.to_pil(image_tensor)

        # Get top 2 texture images (highest scores)
        top_images_pil = texture_crop(
            image=image_pil,
            stride=self.stride,
            window_size=self.window_size,
            metric=self.metric,
            position='top',
            n=2,
            drop=self.drop
        )

        # Get bottom 2 texture images (lowest scores)
        # texture_crop with position='bottom' and n=2 returns [2nd_lowest, lowest]
        # because the internal list `images` is sorted descending by score.
        bottom_images_pil = texture_crop(
            image=image_pil,
            stride=self.stride,
            window_size=self.window_size,
            metric=self.metric,
            position='bottom',
            n=2,
            drop=self.drop
        )

        # Assign based on score:
        # top_images_pil[0] is highest score
        # top_images_pil[1] is second highest score
        img_maxmax_pil = top_images_pil[0]
        img_maxmax1_pil = top_images_pil[1]

        # bottom_images_pil[0] is second lowest score
        # bottom_images_pil[1] is lowest score
        img_minmin_pil = bottom_images_pil[1]    # Lowest score
        img_minmin1_pil = bottom_images_pil[0]   # Second lowest score

        # Convert to tensors
        img_minmin = self.to_tensor(img_minmin_pil)
        img_maxmax = self.to_tensor(img_maxmax_pil)
        img_minmin1 = self.to_tensor(img_minmin1_pil)
        img_maxmax1 = self.to_tensor(img_maxmax1_pil)
        
        return img_minmin, img_maxmax, img_minmin1, img_maxmax1
