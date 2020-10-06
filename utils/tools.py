"""
tools to combine dictionary
"""
import collections
import numpy as np
import cv2

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def Myresize(img, size):
        # print(img.shape)
        h,w = img.shape[:2]
        if h < size[0] or w < size[1]:
            new_h = size[0]
            new_w = size[1]
            h = new_h
            w = new_w
            img = cv2.resize(img,(new_w, new_h))
            
        new_h, new_w = size
        try:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
        except:
            print(h,new_h,w,new_w)
            raise 
        img = img[top: top + new_h,
                            left: left + new_w]
        return img

# torch
# from utils.tools import squeezeToNumpy
def squeezeToNumpy(tensor_arr):
    return tensor_arr.detach().cpu().numpy().squeeze()