from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import subprocess
import os
from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct

app = FastAPI()

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

