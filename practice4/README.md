## FastAPI API

This is a FastAPI-based web service designed for various manipulation tasks of arrays, images and videos.
The application is containerized using Docker. And displayed in a GUI interface using PyQt5.

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
* Convert any input video into VP8, VP9, h265 & AV1.
* Outputs an Encoding Ladder

## How To Use

To clone and run this application, you'll need: 
```bash
# Clone this repository
$ git clone https://github.com/bjporu/SCAV2024

# Go into the repository
$ cd SCAV2024

# Go into the practice folder
$ cd practice4

#REMEMBER TO OPEN DOCKER DESKTOP AND GET IT RUNNING https://docs.docker.com/desktop

#Build the docker as:
$ docker build -t fastapi-app .     

#Run the docker and connect it to SCAV2024 (with your own path) to obatin access the video bbb.mp4 in LAB1 VIDEO. All resizing or Black and White operation results will be stored in that same folder.

$ docker run -d -p 8000:8000 -v /.../SCAV2024/practice3\ VIDEO:/app/images fastapi-app


#To access the GUI 
$ python app_gui.py

# Then select the desired API endpoint from the menu.
# Once selected input the necessary fields and submit. 
# The result will be stored in the Local Folder where you retrieved the image or video from
# specified in /path_to_local_folder:/app/images.
