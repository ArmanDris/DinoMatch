from typing import Dict
from PIL import Image
import os
import pickle
import numpy as np

class RgbImageEmbedder():
    """
    This class stores a dictionary of RGB image embeddings.
    The image name is they key and the RGB tuple is they value.
    """
    def __init__(self, src_folder, dst_file):
        self.data: Dict[str, np.ndarray] = {}
        self.embed_directory_and_save(src_folder, dst_file)

    def print(self) -> None:
        for key, value in self.data.items():
            print(f"{key}:{value}")

    def embed_directory_and_save(self, src_folder: str, dst_file: str) -> None:
        for filename in os.listdir(src_folder):
            if filename.endswith('.png'):
                img_path = os.path.join(src_folder, filename)
                self.data[filename] = self._rgb_encoder(img_path)

        self.save_as(dst_file=dst_file)

    def save_as(self, dst_file: str) -> None:
        try:
            with open(dst_file, "wb") as f:
                pickle.dump(self.data, f)
            print(f" - Saved {len(self.data)} embeddings")
        except OSError as e:
            print(f"Error saving file {dst_file} - {e}")

    """
    Calculates the avg rgb values of an image
    
    Args: 
        image_path (string): Path to the image.
        
    Returns: 
        np.ndarray: a NumPy array containing the average RGB values (r, g, b) each of type uint8.
    """
    def _rgb_encoder(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')

            red_sum, blue_sum, green_sum = 0, 0, 0
        
            pixels = img.getdata()
            for p in pixels:
                r, g, b = p
                red_sum += r
                green_sum += g
                blue_sum += b

            avg_red = red_sum // (img.height * img.width)
            avg_green = green_sum // (img.height * img.width)
            avg_blue = blue_sum // (img.height * img.width)
        
            return np.array([avg_red, avg_green, avg_blue], dtype=np.uint8)
        except OSError as e:
            print(f"Error opening image {image_path} - {e} ")
            return np.array([0, 0, 0], dtype=np.uint8)
    

"""
Saves a dictionary of image embeddings to a file.
The dictionary maps image filenames to their average RGB values.

Args:
    src_folder (str, optional): The directory containing the images. Defaults to './data/10_Demo_Images'.
    dst_file (str, optional): What the file will be saved as. Defaults to ./embeddings/RGB_embeddings.json.
    print_embeddings (bool, optional): Whether or not to print embeddings to console. Defaults to False.

Returns:
    None
        
"""
def save_rgb_embeddings(src_folder: str = "./data/10_Demo_Images", dst_file: str = "./data/embeddings/RGB_embeddings.pickle", print_embeddings: bool = False) -> None:

    collection = RgbImageEmbeddings()
    
    for filename in os.listdir(src_folder):
        if filename.endswith('.png'):
            collection.add_image(filename, os.path.join(src_folder, filename))

    if print_embeddings:
        collection.print()

    collection.save_as(dst_file)

if __name__ == "__main__":
    save_rgb_embeddings(print_embeddings = True)
