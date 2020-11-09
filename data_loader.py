import numpy as np
import glob
from skimage.transform import resize

from utils.img_ops import *

# Helper functions for reading keypoint data
def get_files(file_ext=".asf"):
    """Gets list of files from imm face db with correct face type

    Args:
        type (int, optional): Type number. Defaults to 1.

    Returns:
        [list]: list of file paths
    """
    files = glob.glob(f"imm_face_db/*{file_ext}")
    files.sort()
    return files

def read_asf(file, keypoints):
    """Reads x, y points from asf file

    Args:
        file (str): path to asf file

    Returns:
        np.ndarray, np.ndarray: list of x points, list of y points
    """
    data = np.genfromtxt(file, skip_header=16, skip_footer=1, usecols=(2, 3))[keypoints, :]
    return data[:, 0], data[:, 1]

def format_points(x, y, width, height):
    """Unnormalizes points and Formats as a numpy array

    Args:
        x ([arr])
        y ([arr])

    Returns:
        np.ndarray with size + 4
    """
    x, y = x * width, y * height
    return np.column_stack((x, y))

def rescale_and_center_crop(img, shape):
    scale = shape[0] / img.shape[0]
    img = resize(img,(img.shape[0] * scale, int(img.shape[1] * scale)))
    
    if img.shape[1] == shape[1]:
        return img
    
    half = (img.shape[1] - shape[1]) / 2
    i = int(img.shape[1] - shape[1] - half)
    j = int(img.shape[1] - shape[1] - i)
    img = img[:, i:-j]
    
    return img

def load_data(val_split=(32, 8), keypoints=[-6]):
    """Loads and splits faces data as numpy arrays

    Args:
        val_split (tuple, optional): (train frac, validation frac)
                                     Represents ratio between train and validation data.
                                     Defaults to (32, 8).

        keypoints (int, optional): Number of keypoints to return. Defaults to the nose keypoint.
    """
    height, width = 60, 80

    # Get file names
    img_files = get_files(file_ext='.jpg')
    asf_files = get_files(file_ext='.asf')

    # Read files and format
    imgs = np.array([rescale_and_center_crop(read(i), (height, width)) for i in img_files])
    asf_data = [read_asf(f, keypoints) for f in asf_files]
    asf_data = np.row_stack([format_points(x[0], x[1], 1, 1) for x in asf_data])

    train_ratio = val_split[0] / (val_split[0] + val_split[1])
    train_imgs, val_imgs = np.split(imgs, [int(train_ratio * imgs.shape[0])], axis=0)
    train_asf, val_asf = np.split(asf_data, [int(train_ratio * imgs.shape[0])], axis=0)

    return train_imgs, train_asf, val_imgs, val_asf
