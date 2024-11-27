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

# Data model for resolution change
class ResolutionRequest(BaseModel):
    input_video: str
    output_video: str
    width: int
    height: int
    
class ChromaSubsamplingRequest(BaseModel):
    input_video: str
    output_video: str
    pixel_format: str
    
class VideoInfo(BaseModel):
    input_video: str

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

#SCAV S2

#EXERCISE 1: Create a new endpoint / feature which will let you to modify the resolution (use FFmpeg in the backend)

@app.post("/change_resolution/")
def change_resolution(data: ResolutionRequest):
    # Ensure the video exists
    if not os.path.exists(data.input_video):
        raise HTTPException(status_code=400, detail="Input video path not found")
    # Run FFmpeg to change resolution
    command = [
        "ffmpeg",
        "-i", data.input_video,
        "-vf", f"scale={data.width}:{data.height}",
        data.output_video
    ]
    try:
        subprocess.run(command, check=True)
        return {"message": "Resolution changed successfully", "output_file": data.output_video}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="Failed to change video resolution.")


#EXERCISE 2: 2) Create a new endpoint / feature which will let you to modify the chroma subsampling

@app.post("/change_chroma_subsampling/")
def change_chroma_subsampling(data: ChromaSubsamplingRequest):
    # Ensure the input video exists
    if not os.path.exists(data.input_video):
        raise HTTPException(status_code=400, detail="Input video path not found")

    # FFmpeg command to change chroma subsampling
    command = [
        "ffmpeg",
        "-i", data.input_video,
        "-c:v", "libx264",
        "-pix_fmt", data.pixel_format,
        data.output_video
    ]
    try:
        subprocess.run(command, check=True)
        return {"message": "Chroma subsampling changed successfully", "output_file": data.output_video}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to change chroma subsampling.")

#EXERCISE 3:  Create a new endpoint / feature which lets you read the video info and print at least 5 relevant data from the video

@app.post("/get_video_info/")
def get_video_info(data: VideoInfo):

    # Check if the file exists
    if not os.path.exists(data.input_video):
        raise HTTPException(status_code=404, detail="File not found")

    # FFmpeg command to extract metadata
    command = [
        "ffmpeg",
        "-i", data.input_video,
        "-hide_banner"
    ]

    try:
        # Run the FFmpeg command and capture the output
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

        # Parse the FFmpeg output
        output = result.stderr

        # Initialize fields with default values
        duration = resolution = bitrate = framerate = videocodec = audiocodec = containerformat = "Unknown"

        duration = output.split("Duration:")[1].split(",")[0].strip()

        # Extract resolution (Width x Height)
        video_info = output.split("Video:")[1].split(",")
        resolution_line = [item.strip() for item in video_info if "x" in item and "fps" not in item]
        resolution = resolution_line[0].split(" ")[0]

        # Extract bitrate
        bitrate = output.split("bitrate:")[1].split("\n")[0].strip()

        # Extract frame rate (fps)
        framerate_line = [item for item in video_info if "fps" in item]
        framerate = framerate_line[0].strip()

        # Extract video codec
        videocodec = output.split("Video:")[1].split(",")[0].split()[0]

        # Extract audio codec
        audio_info = output.split("Audio:")[1].split(",")
        audiocodec = audio_info[0].strip()

        # Extract container format
        container_info = output.split("Input #")[1].split(":")[0]
        containerformat = container_info.strip()

        # Return the parsed information
        return {
            "Duration": duration,
            "Resolution": resolution,
            "Bitrate": bitrate,
            "Frame Rate": framerate,
            "Video Codec": videocodec,
            "Audio Codec": audiocodec,
            "Container Format": containerformat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve video info: {str(e)}")
