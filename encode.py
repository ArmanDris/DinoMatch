import numpy as np
from PIL import Image
import os

# Encodes an RGB image as a numpy array: [R, G, B]
def basicRgbEncoder(frame_path):
    frame = Image.open(frame_path)
    frame = frame.convert('RGB')
    np_frame = np.array(frame)
    average_color = np_frame.mean(axis=(0, 1))
    return average_color

def encodeFolder(folder_path = './data/10_Demo_Images'):
    frame_arrays = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            frame_embedding = basicRgbEncoder(folder_path + '/' + filename)
            frame_arrays.append(frame_embedding)

    if not frame_arrays:
        print("No PNG images found")
        return np.array([])
    
    np_frame_arrays = np.array(frame_arrays)
    print("Encoded " + str(np_frame_arrays.shape[0]) + " images")
    np.save("embeddings/frames.npy", np_frame_arrays)


encodeFolder()