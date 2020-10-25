import numpy as np
import glob

# Helper functions for reading keypoint data
def get_by_face_type(type=1, file_ext='.asf'):
    """Gets list of files from imm face db with correct face type

    Args:
        type (int, optional): Type number. Defaults to 1.

    Returns:
        [list]: list of file paths
    """
    files = glob.glob(f'in/imm_face_db/*-{type}m{file_ext}')
    files.sort()
    return files

def read_asf(file):
    """Reads x, y points from asf file

    Args:
        file (str): path to asf file

    Returns:
        np.ndarray, np.ndarray: list of x points, list of y points
    """
    data = np.genfromtxt(file, skip_header=16, skip_footer=1, usecols=(2, 3))
    return data[:, 0], data[:, 1]

def format_points(x, y, width, height):
    """Unnormalizes points, Adds corner points, and Formats as a numpy array

    Args:
        x ([arr])
        y ([arr])
    
    Returns:
        np.ndarray with size + 4
    """
    x, y = x * width, y * height
    x = np.append(x, [1, 1, width - 1, width - 1])
    y = np.append(y, [1, height - 1, 1, height - 1])
    return np.column_stack((x, y))
