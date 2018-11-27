#!/usr/bin/env python
'''Pre-processing code:
This code reads all frames for a video to either a torch tensor or numpy array.

create 10fps And store then in a directory using an 'image_tmpl' for the naming.
for 10 fps: ffmpeg -i test-news.mp4 -vf fps=10 image-%03d.png
    
    
Take into account that this code might run different on Windows or Linux ! '''
# coding: utf-8
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
from skimage.transform import resize
from scipy.ndimage import filters
from torch.autograd import Variable
from torchvision import transforms
imageio.plugins.ffmpeg.download() #install ffmpeg plugin

#   - Numpy -

def _load_image(directory, image_tmpl, idx, color):
    '''Loads an image from file with PIL and converts it to either "RGB" or "LA"'''
    try:
        return Image.open(os.path.join(directory, image_tmpl.format(idx))).convert(color)
    except IOError:
        print('error loading image:', os.path.join(directory, image_tmpl.format(idx)))

def _to_np_array(pic):
    return np.array(pic).astype(np.float_)

def normalize(fr_arr, color):
    '''normalizes a np.array with RGB or LA.'''
    if color == 'RGB':
        for i in range(3):
            minval = fr_arr[..., i].min()
            maxval = fr_arr[..., i].max()
            if minval != maxval:
                fr_arr[..., i] -= minval
                fr_arr[..., i] *= (255.0/(maxval-minval))
    if color == 'LA':
        cv2.normalize(fr_arr, fr_arr, 0, 255, cv2.NORM_MINMAX)
    return fr_arr

def gaussian_blur(frame, color):
    '''apply gausian blur to either RGB or LA np.array'''
    if color == 'RGB':
        for i in range(3):
            frame[:, :, i] = filters.gaussian_filter(frame[:, :, i], 5)
    if color == 'LA':
        frame = filters.gaussian_filter(frame, 5)
    return frame

def frames_to_float(directory, image_tmpl, color):
    '''converts all frames to a np.array with float numbers.'''
    fr_list = []
    fr_amount = len(next(os.walk(directory))[2]) #counts the amount of frames 
    indices = np.asarray(range(1, fr_amount))
    for idx in indices:
        fr_i = _load_image(directory, image_tmpl, idx, color) #load single image
        fr_i = _to_np_array(fr_i)
        if color == 'RGB':
            fr_i = resize(fr_i, (700, 1280, 3))
        elif color == 'LA':
            fr_i = resize(fr_i, (700, 1280, 2))
        fr_i = normalize(fr_i, color)
        fr_i = gaussian_blur(fr_i, color)
        fr_list.append(fr_i)
    return fr_amount, fr_list

#   - Torch Tensor -

def frames_to_tensor(directory, image_tmpl, import_idx):
    '''converts frame images to tensor. Resizes to 224x224 and normalizes.'''
    tensor_list = []
    if len(import_idx) == 2:  #import only a fragment
        indices = np.asarray(range((import_idx[0] + 1), (import_idx[1] + 1)))
    elif import_idx == 'all': #import whole video
        fr_amount = len(next(os.walk(directory))[2]) #counts the amount of frames
        indices = np.asarray(range(1, fr_amount)) #first had +1
    #define trasformation
    preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    for idx in indices:
        fr_i = _load_image(directory, image_tmpl, idx, 'RGB') #load single image
        fr_i = Variable(preprocess(fr_i).unsqueeze(0)) #create tensor with transforms
        tensor_list.append(fr_i)
    return len(indices), tensor_list

#   - Preprocess single fram -

def process_single_fr(file_path, frame_format):
    '''goes from file path to either:
    - a gray np array(frame_format= "LA") or
    - a RGB np array (frame_format = "RGB")
    - a torch tensor (frame_format = "tensor")'''

    if frame_format == 'LA':
        frame = Image.open(file_path).convert(frame_format)
        frame = _to_np_array(frame)
        frame = resize(frame, (700, 1280, 2))
        frame = normalize(frame, frame_format)
        frame = gaussian_blur(frame, frame_format)
    if frame_format == 'RGB':
        frame = Image.open(file_path).convert(frame_format)
        frame = _to_np_array(frame)
        frame = resize(frame, (700, 1280, 3))
        frame = normalize(frame, frame_format)
        frame = gaussian_blur(frame, frame_format)
    if frame_format == 'tensor':
        preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        frame = Image.open(file_path).convert('RGB')
        frame = Variable(preprocess(frame).unsqueeze(0)) #create tensor with transforms
    return frame

#   - Visualizing -

def show_fr_list(fr_list, filename):
    '''Given any list of frames it plots the images in a raster
    and saves them to a png file.'''
    plt.figure(figsize=(20, 20))
    columns = 8
    for fr_nr, frame in enumerate(fr_list):
        pil_im = Image.fromarray(frame.astype('uint8'))
        plt.subplot(len(fr_list) / columns + 1, columns, fr_nr + 1)
        plt.title('Frame #{}'.format(fr_nr+1), fontsize=12)
        plt.imshow(pil_im, cmap=plt.cm.gray)#, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(filename)
