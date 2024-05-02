import numpy as np
from PIL import Image
import os
import json

def save_as_embeddings(image_folder = './data/10_Demo_Images'):

    # Encodes an RGB image as a numpy array: [R, G, B]
    def basicRgbEncoder(frame_path):
        frame = Image.open(frame_path)
        frame = frame.convert('RGB')
        np_frame = np.array(frame)
        average_color = np_frame.mean(axis=(0, 1))
        return average_color

    def encodeFolder():
        frame_arrays = []
        frame_names = []
        for filename in os.listdir(image_folder):
            if filename.endswith('.png'):
                frame_names.append(filename)
                frame_embedding = basicRgbEncoder(image_folder + '/' + filename)
                frame_arrays.append(frame_embedding)


        if not frame_arrays or not frame_names:
            print("No images found")
            return np.array([])
        
        np_frame_arrays = np.array(frame_arrays)
        json_data = [{"name": name} for name in frame_names]
        np.save("embeddings/frames.npy", np_frame_arrays)
        with open('embeddings/names.json', 'w') as json_file:
            for item in json_data:
                json_string = json.dumps(item)
                json_file.write(json_string + '\n')
        print(" - Encoded " + str(len(frame_arrays)) + " images and " + str(len(frame_names)) + " vector names")
        
    encodeFolder()

if __name__ == "__main__":
    save_as_embeddings()