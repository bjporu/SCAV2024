## FastAPI API

This is a FastAPI-based web service designed for various image and data processing tasks.
The application is containerized using Docker.

## Objectives 
The primary objectives of this project were:
1. **Develop a FastAPI-based API** for image and signal processing.
2. **Implement color space conversion** for RGB to YUV and YUV to RGB.
3. **Provide image manipulation capabilities** such as resizing and black & white conversion.
4. **Enable signal processing features** such as DCT encoding and decoding, Run-Length Encoding (RLE), and Serpentine scanning.
5. **Containerize the application** using Docker to ensure scalability and portability.
6. **Use FFmpeg** to handle image resizing and black-and-white conversion tasks.

## Key Features

Key Features

* 1. Image Processing
- RGB to YUV Conversion: Convert RGB values to YUV color space.
- YUV to RGB Conversion: Convert YUV values back to RGB.
- Resize Image: Resize an image to specified dimensions using FFmpeg.
- Convert to Black and White: Convert an image to grayscale using FFmpeg.
* 2. Signal Processing
- DCT (Discrete Cosine Transform):
- Encode data using the DCT.
- Decode DCT-encoded data back to the original signal.
- RLE (Run-Length Encoding):
- Encode data using RLE for compression.
- Decode RLE-encoded data back to the original sequence.
* 3. Serpentine (Zigzag) Scan
_ Perform a serpentine scan (Zigzag scan) on a 2D matrix for data processing or compression.

## How To Use

To clone and run this application, you'll need: 
```bash
# Clone this repository
$ git clone https://github.com/bjporu/SCAV2024

# Go into the repository
$ cd SCAV2024

# Go into the practice folder
$ cd practice1

#REMEMBER TO OPEN DOCKER DESKTOP AND GET IT RUNNING https://docs.docker.com/desktop

#Build the docker as:
$ docker build -t fastapi-app .     

#Run the docker and connect it to SCAV2024 (with your own path) to obatin access the image snoop_dogg.jpeg in LAB1 VIDEO. All resizing or Black and White operation results will be stored in that same folder.

$ docker run -d -p 8000:8000 -v /.../SCAV2024/LAB1\ VIDEO:/app/images fastapi-app


#To access the api you can go to your browser of preference and input the following link

# http://localhost:8000/docs
# 127.0.0.1:8000/docs
# 0.0.0.0:8000/docs #Might not work due to firewalls

# From there you will be able to see and interact with all endpoints by clicking on "Try it out".
# However you can always call them from the terminal as:
$ curl -X 'POST' \
  'http://localhost:8000/rgb_to_yuv/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "R": 100,
  "G": 200,
  "B": 120
}'

#IMPORTANT! WHEN USING FFMPEG RELATED ENDPOINTS (resize and black_and_white)
THe input and output paths must be declared as: /app/images
WHY?
Remember? 
$ docker run -d -p 8000:8000 -v /.../SCAV2024/LAB1\ VIDEO:/app/images fastapi-app
That's why

To stop the containers you simply have to run:
$ docker ps

# You will see the CONTAINER ID, copy it and run the following command:
$ docker stop CONTAINER_ID

```
