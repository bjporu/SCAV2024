import os
import sys

from fastapi import FastAPI, HTTPException

import numpy as np

from pydantic import BaseModel
from typing import List, Dict
import subprocess

import json
import time


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
    
class BlackandWhiteRequest(BaseModel):
    input_image: str
    output_image: str
    
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
    
class VideoInfo2(BaseModel):
    input_video: str
    output_path: str
    
class VideoMacroMotion(BaseModel):
    input_video: str
    output_video: str

class YUVHistogramRequest(BaseModel):
    input_video: str
    output_video: str

class VideoConversionInput(BaseModel):
    input_file: str
    output_file: str
    codec: str

class EncodingLadderRequest(BaseModel):
    input_video: str  # Input video path
    output_video: str  # Base output file name
    codecs: List[str]  # List of codecs to convert to (e.g., vp8, vp9, h265, av1)
    resolutions: List[Dict[str, int]]



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
def convert_to_black_and_white(data: BlackandWhiteRequest):
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
    # We can reuse the command from the first lab which allowed us to do the resizing procedure
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
# https://trac.ffmpeg.org/wiki/Chroma%20Subsampling

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
        "-vf", f"format={data.pixel_format}",
        data.output_video
    ]
    try:
        subprocess.run(command, check=True)
        return {"message": "Chroma subsampling changed successfully", "output_file": data.output_video}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to change chroma subsampling.")

#EXERCISE 3:  Create a new endpoint / feature which lets you read the video info and print at least 5 relevant data from the video
# https://ffmpeg.org/ffmpeg.html#Main-options
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

        # Return the parsed information
        return {
            "Duration": duration,
            "Resolution": resolution,
            "Bitrate": bitrate,
            "Frame Rate": framerate,
            "Video Codec": videocodec,
            "Audio Codec": audiocodec,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve video info: {str(e)}")

#EXERCISE 4: ) You’re going to create another endpoint in order to create a new BBB container. It will fulfill this requirements:
#· Cut BBB into 20 seconds only video.
#· Export BBB(20s) audio as AAC mono track.
#· Export BBB(20s) audio in MP3 stereo w/ lower bitrate
#· Export BBB(20s) audio in AC3 codec
# Now package everything in a .mp4 with FFMPEG!! - DO NOT UPLOAD VIDEOS INTO GITHUB

@app.post("/create_bbb_container/")
def create_bbb_container(data: VideoInfo2):
    # Check if the input video file exists
    if not os.path.exists(data.input_video):
        raise HTTPException(status_code=404, detail="Input video not found")

    # Define the output file paths
    
    output_video = os.path.join(data.output_path, "bbb_20s.mp4")
    output_audio_aac = os.path.join(data.output_path, "bbb_20s_aac.m4a")
    output_audio_mp3 = os.path.join(data.output_path, "bbb_20s_mp3.mp3")
    output_audio_ac3 = os.path.join(data.output_path, "bbb_20s_ac3.ac3")
    
    # FFmpeg command to trim the video to 20 seconds and export audio
    try:
        # Trim the video to 20 seconds and export it as a new video file
        # https://ffmpeg.org/ffmpeg.html#Time
        # -c:v and -c:a specify the video and audio codec respectively
        command_video = [
            "ffmpeg", "-i", data.input_video, "-t", "20", "-c:v", "libx264", "-c:a", "aac",
            "-b:a", "192k", "-y", output_video
        ]
        subprocess.run(command_video, check=True)
        # Extract AAC mono audio https://stackoverflow.com/questions/47087802/ffmpeg-how-to-convert-audio-to-aac-but-keep-bit-rate-at-what-the-old-file-used
        # https://www.mux.com/articles/change-video-bitrate-with-ffmpeg
        command_aac = [
            "ffmpeg", "-i", data.input_video, "-t", "20", "-vn", "-acodec", "aac", "-ac", "1", "-b:a", "192k",
            "-y", output_audio_aac
        ]
        subprocess.run(command_aac, check=True)

        # Extract MP3 stereo audio with lower bitrate
        # https://stackoverflow.com/questions/38449239/converting-all-the-mp4-audio-files-in-a-folder-to-mp3-using-ffmpeg
        # https://www.mux.com/articles/change-video-bitrate-with-ffmpeg
        command_mp3 = [
            "ffmpeg", "-i", data.input_video, "-t", "20", "-vn", "-acodec", "libmp3lame", "-ac", "2", "-b:a", "96k",
            "-y", output_audio_mp3
        ]
        subprocess.run(command_mp3, check=True)

        # Extract AC3 audio https://superuser.com/questions/1279589/using-ffmpeg-to-convert-to-ac3-and-remove-extra-audio-tracks
        # This command takes the first 20 seconds of bbb.mp4 and encodes it as AC3 at 192k bitrate, and saves it to the specified file.
        command_ac3 = [
            "ffmpeg", "-i", data.input_video, "-t", "20", "-vn", "-acodec", "ac3", "-b:a", "192k", "-y", output_audio_ac3
        ]
        subprocess.run(command_ac3, check=True)

        # Now, package everything into a .mp4 file with the video and audio tracks https://trac.ffmpeg.org/wiki/Map
        #This command basically merges a video and three audio tracks (AAC, MP3, AC3) into an MP4 file. It ensures the video is untouched (`copy` codec), assigns each audio codec to specific tracks, maps video and audio streams
        #correctly, and overwrites existing outputs. The final file consolidates all formats in final_bbb_container.mp4.
        command_final = [
            "ffmpeg", "-i", output_video, "-i", output_audio_aac, "-i", output_audio_mp3, "-i", output_audio_ac3,
            "-c:v", "copy", "-c:a:0", "aac", "-c:a:1", "libmp3lame", "-c:a:2", "ac3", "-map", "0:v", "-map", "1:a",
            "-map", "2:a", "-map", "3:a", "-y", os.path.join(data.output_path, "final_bbb_container.mp4")
        ]
        subprocess.run(command_final, check=True)

        return {"message": "BBB container created successfully", "output_file": "final_bbb_container.mp4"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during video processing: {e}")
        
        

#EXERCISE 5: Create a new endpoint / feature which reads the tracks from an MP4 container, and it’s able to say (deliver an output) of how many tracks does the container contains
@app.post("/video_tracks/")
def video_tracks(data: VideoInfo):
    # Check if the input video exists
    if not os.path.isfile(data.input_video):
        raise HTTPException(status_code=404, detail="The specified video file could not be found.")

     # Build the ffprobe command to extract stream details from the video
    command = f"ffprobe -v quiet -show_entries stream=index,codec_name,codec_type -of json {data.input_video}"
    
    try:
        # Execute the ffprobe command and parse the output
        output = subprocess.check_output(command, shell=True, text=True)
        video_data = json.loads(output)  # Convert JSON string to Python object

        # Extract stream information
        tracks = video_data.get("streams", [])
        num_tracks = len(tracks)

        # Build the response
        track_details = [
            {"index": track["index"], "codec_type": track["codec_type"], "codec_name": track["codec_name"]}
            for track in tracks
        ]
        return {"num_tracks": num_tracks, "tracks": track_details}
    
    except subprocess.CalledProcessError as e:
        # Handle ffprobe execution errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video file: {e}")
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        raise HTTPException(status_code=500, detail="Failed to parse ffprobe output.")


#EXERCISE 6: Create a new endpoint / feature which will output a video that will show the macroblocks and the motion vectors
@app.post("/visualize_motion_vectors/")
def visualize_macroblocks_motion_vectors(data: VideoMacroMotion):
    # Check if the input video exists
    if not os.path.isfile(data.input_video):
        raise HTTPException(status_code=404, detail="The specified input video file could not be found.")

    # Build the ffmpeg command for visualizing motion vectors: https://trac.ffmpeg.org/wiki/Debug/MacroblocksAndMotionVectors
    command = f"ffmpeg -hide_banner -flags2 +export_mvs -i {data.input_video} -vf codecview=mv=pf+bf+bb -an {data.output_video}"
    
    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)

        # Return success message
        return {
            "message": "Motion vectors visualized successfully.",
            "output_video": data.output_video
        }
    except subprocess.CalledProcessError as e:
        # Handle ffmpeg execution errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {e}")
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


#EXERCISE 7: from fastapi import FastAPI, HTTPException

@app.post("/generate_yuv_histogram/")
def generate_yuv_histogram(data: YUVHistogramRequest):
    # Check if the input video exists
    if not os.path.isfile(data.input_video):
        raise HTTPException(status_code=404, detail="The specified input video file could not be found.")

    # Build the ffmpeg command to generate a video with the YUV histogram
    command = (
        f'ffmpeg -hide_banner -i {data.input_video} -vf '
        f'"split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay" {data.output_video}'
    )
    
    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)

        # Return success message
        return {
            "message": "YUV histogram video generated successfully.",
            "output_video": data.output_video
        }
    except subprocess.CalledProcessError as e:
        # Handle ffmpeg execution errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {e}")
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

#EXERCISE 7: Create a new endpoint / feature which will output a video that will show the YUV histogram

@app.post("/generate_yuv_histogram/")
def generate_yuv_histogram(data: YUVHistogramRequest):
    # Check if the input video exists
    if not os.path.isfile(data.input_video):
        raise HTTPException(status_code=404, detail="The specified input video file could not be found.")

    # Build the ffmpeg command to generate a video with the YUV histogram https://trac.ffmpeg.org/wiki/Histogram
    command = (
        f'ffmpeg -hide_banner -i {data.input_video} -vf '
        f'"split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay" {data.output_video}'
    )
    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)

        # Return success message
        return {
            "message": "YUV histogram video generated successfully.",
            "output_video": data.output_video
        }
    except subprocess.CalledProcessError as e:
        # Handle ffmpeg execution errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the video: {e}")
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# PRACTICE 4

# EXERCISE 1: Create a new endpoint/feature to convert any input video into VP8, VP9, h265 & AV1.
  
    # References we used to generate the command for every codec.
    #"vp8": "libvpx" https://trac.ffmpeg.org/wiki/Encode/VP8
    #"vp9": "libvpx-vp9" https://trac.ffmpeg.org/wiki/Encode/VP9
    #"h265": "libx265" https://trac.ffmpeg.org/wiki/Encode/H.265
    #"av1": "libaom-av1" https://trac.ffmpeg.org/wiki/Encode/AV1


@app.post("/convert-video/")
def convert_video(input_data: VideoConversionInput):
    if not os.path.exists(input_data.input_file):
        raise HTTPException(status_code=400, detail="Input file does not exist")

    try:
        # Define output filename
        output_file = f"{os.path.splitext(input_data.input_file)[0]}_{input_data.codec.lower()}"
        if input_data.codec.lower() == "vp8":
            cmd = f'ffmpeg -i "{input_data.input_file}" -c:v libvpx -b:v 0 "{output_file}.webm"'
        elif input_data.codec.lower() == "vp9":
            cmd = f'ffmpeg -i "{input_data.input_file}" -c:v libvpx-vp9 -b:v 0 "{output_file}.webm"'
        elif input_data.codec.lower() == "h265":
            cmd = f'ffmpeg -i "{input_data.input_file}" -c:v libx265 -b:v 0 "{output_file}.mp4"'
        elif input_data.codec.lower() == "av1":
            cmd = f'ffmpeg -i "{input_data.input_file}" -c:v libaom-av1 -b:v 0 "{output_file}.mkv"'
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported codec: {input_data.codec}")

        # Execute the command
        subprocess.run(cmd, shell=True, check=True)

        return {
            "message": f"Conversion to {input_data.codec} completed successfully.",
            "output_file": f"{output_file}.{input_data.codec.lower()}"
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during conversion: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# EXERCISE 2: Create a new endpoint/feature to be able to do an encoding ladder.

@app.post("/encoding-ladder/")
def encoding_ladder(data: EncodingLadderRequest):
    if not os.path.exists(data.input_video):
        raise HTTPException(status_code=400, detail="Input video file does not exist")

    output_files = []

    # Apply resolution changes first
    for resolution in data.resolutions:
        resolution_output_file = f"{os.path.splitext(data.input_video)[0]}_{resolution['width']}x{resolution['height']}"

        resolution_data = ResolutionRequest(
            input_video=data.input_video,
            width=resolution["width"],
            height=resolution["height"],
            output_video=f"{resolution_output_file}.mp4"
        )

        try:
            resolution_result = change_resolution(resolution_data)
            output_files.append(resolution_result["output_file"])
        except HTTPException as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"Error changing resolution to {resolution['width']}x{resolution['height']}: {e.detail}"
            )

    # After applying resolution changes, convert each resolution to different codecs
    final_output_files = []
    for resolution_output in output_files:
        for codec in data.codecs:
            codec_data = VideoConversionInput(
                input_file=resolution_output,
                output_file=resolution_output,
                codec=codec
            )

            try:
                conversion_result = convert_video(codec_data)
                final_output_files.append(conversion_result["output_file"])
            except HTTPException as e:
                raise HTTPException(
                    status_code=e.status_code,
                    detail=f"Error converting {resolution_output} to {codec}: {e.detail}"
                )

    return {
        "message": "Encoding ladder completed successfully.",
        "output_files": final_output_files
    }
