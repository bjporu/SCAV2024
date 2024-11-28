## FastAPI API

This is a FastAPI-based web service designed for various manipulation tasks of the Big Buck Bunny video.
The application is containerized using Docker.

## Key Features

* Modify the resolution
* Modify the chroma subsampling
* Extract the rellevant data from the video file
* Crop the video into a 20-seconds section
* Export its audio as an ACC mono track
* Export its audio in mp3 stereo
* Export its audio in AC3 codec
* Reads the tracks from an MP4 container, such that it’s able to deliver an output how many tracks does the container contains
* Output a video that shows the macroblocks and the motion vectors
* Output a video that shows the the YUV histogram

## How To Use

To clone and run this application, you'll need: 
```bash
# Clone this repository
$ git clone https://github.com/bjporu/SCAV2024

# Go into the repository
$ cd SCAV2024

# Go into the practice folder
$ cd practice3

#REMEMBER TO OPEN DOCKER DESKTOP AND GET IT RUNNING https://docs.docker.com/desktop

#Build the docker as:
$ docker build -t fastapi-app .     

#Run the docker and connect it to SCAV2024 (with your own path) to obatin access the video bbb.mp4 in LAB1 VIDEO. All resizing or Black and White operation results will be stored in that same folder.

$ docker run -d -p 8000:8000 -v /.../SCAV2024/practice3\ VIDEO:/app/images fastapi-app


#To access the api you can go to your browser of preference and input the following link

# http://localhost:8000/docs
# 127.0.0.1:8000/docs
# 0.0.0.0:8000/docs #Might not work due to firewalls

# From there you will be able to see and interact with all endpoints by clicking on "Try it out".
# However you can always call them from the terminal as:
curl -X 'POST' \
  'http://localhost:8000/change_chroma_subsampling/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_video": "/app/images/bbb.mp4",
  "output_video": "/app/images/bbb_chroma_subsampled.mp4",
  "pixel_format": "yuv444p"
}'

#IMPORTANT! WHEN USING FFMPEG RELATED ENDPOINTS
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
