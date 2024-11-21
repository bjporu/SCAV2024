import os
import sys

from fastapi import FastAPI, HTTPException

import numpy as np

from pydantic import BaseModel
from typing import List
import subprocess


from scipy.fftpack import dct, idct

#Initializes the aapp
app = FastAPI()

# Data models
class RGB(BaseModel):
    R: int
    G: int
    B: int

class YUV(BaseModel):
    Y: float
    U: float
    V: float

class ImageRequest(BaseModel):
    input_image: str
    output_image: str
    width: int
    height: int

class DCTRequest(BaseModel):
    type_of_dct: int
    input_data: List[float]

class RLERequest(BaseModel):
    input_data: List[int]

class SerpentineRequest(BaseModel):
    input_data: List[List[int]]

class SerpentineResponse(BaseModel):
    scanned_data: List[int]


# Translator class for image manipulation and color space conversion
class Translator:

    def rgb_to_yuv(self, R, G, B):
        conversion_matrix = np.array([[ 0.299,  0.587, 0.114],
                                      [-0.14713, -0.28886, 0.436],
                                      [0.615,  -0.51499, -0.10001]])
        
        rgb = np.array([R, G, B])
        return np.matmul(conversion_matrix, rgb)

    def yuv_to_rgb(self, Y, U, V):
        conversion_matrix = np.array([[ 1,  0, 1.13983],
                                      [1, -0.39465, -0.58060],
                                      [1,  2.03211, 0]])
        
        yuv = np.array([Y, U, V])
        return np.matmul(conversion_matrix, yuv)

    def resize(self, width, height, input_image, output_image):
        command = [
            'ffmpeg',
            '-i', input_image,
            '-vf', f'scale={width}:{height}',
            '-frames:v', '1',
            output_image
        ]
        subprocess.run(command, check=True)

    def black_and_white(self, input_image, output_image):
        command = [
            'ffmpeg',
            '-i', input_image,
            '-vf', 'format=gray',
            '-qscale:v', '31', output_image
        ]
        subprocess.run(command, check=True)


# Run-Length Encoding (RLE)
def run_length_encode(data):
    encoded = []
    prev_value = data[0]
    count = 1
    for i in range(1, len(data)):
        if data[i] == prev_value:
            count += 1
        else:
            encoded.append((prev_value, count))
            prev_value = data[i]
            count = 1
    encoded.append((prev_value, count))  # Append the last value
    return encoded

def run_length_decode(encoded_data):
    decoded = []
    for value, count in encoded_data:
        decoded.extend([value] * count)
    return decoded

# Serpentine (Zigzag) Scan
def serpentine_scan(input_matrix):
    result = []
    rows = len(input_matrix)
    cols = len(input_matrix[0]) if rows > 0 else 0
    
    # Loop over each diagonal sum (from 0 to rows + cols - 2)
    for s in range(rows + cols - 1):
        diagonal = []  # Temporary list to hold elements of the current diagonal
        
        # Determine the starting point of the diagonal
        if s < rows:
            row = s
            col = 0
        else:
            row = rows - 1
            col = s - row

        # Collect all elements along the diagonal
        while row >= 0 and col < cols:
            diagonal.append(input_matrix[row][col])
            row -= 1  # Move up
            col += 1  # Move right

        # Append the diagonal to result in correct order
        if s % 2 == 0:
            result.extend(diagonal)  # Diagonals with even index in upwards order
        else:
            result.extend(diagonal[::-1])  # Diagonals with odd in downwards order

    return result



# API Endpoints

@app.post("/rgb_to_yuv/")
def convert_rgb_to_yuv(data: RGB):
    translator = Translator()
    YUV = translator.rgb_to_yuv(data.R, data.G, data.B)
    return {"YUV": YUV.tolist()}


@app.post("/yuv_to_rgb/")
def convert_yuv_to_rgb(data: YUV):
    translator = Translator()
    RGB_values = translator.yuv_to_rgb(data.Y, data.U, data.V)
    return {"RGB": RGB_values.tolist()}


@app.post("/resize/")
def resize_image(data: ImageRequest):
    if not os.path.exists(data.input_image):
        raise HTTPException(status_code=400, detail="Input image path not found")
    translator = Translator()
    translator.resize(data.width, data.height, data.input_image, data.output_image)
    return {"message": "Image resized successfully"}


@app.post("/black_and_white/")
def convert_to_black_and_white(data: ImageRequest):
    if not os.path.exists(data.input_image):
        raise HTTPException(status_code=400, detail="Input image path not found")
    translator = Translator()
    translator.black_and_white(data.input_image, data.output_image)
    return {"message": "Image converted to black and white successfully"}


@app.post("/dct/encode/")
def encode_dct(data: DCTRequest):
    dct_data = dct(data.input_data, type=data.type_of_dct)
    return {"encoded_data": dct_data.tolist()}


@app.post("/dct/decode/")
def decode_dct(data: DCTRequest):
    decoded_data = idct(data.input_data, type=data.type_of_dct, norm='ortho')
    return {"decoded_data": decoded_data.tolist()}


@app.post("/rle/encode/")
def encode_rle(data: RLERequest):
    encoded = run_length_encode(data.input_data)
    return {"encoded_data": encoded}


@app.post("/rle/decode/")
def decode_rle(data: RLERequest):
    decoded = run_length_decode(data.input_data)
    return {"decoded_data": decoded}

@app.post("/serpentine/")
def serpentine_scan_data(data: SerpentineRequest):
    # Check if the input data is valid (not empty)
    if not data.input_data:
        raise HTTPException(status_code=400, detail="Input data cannot be empty.")
    
    scanned_data = serpentine_scan(data.input_data)
    return SerpentineResponse(scanned_data=scanned_data)


