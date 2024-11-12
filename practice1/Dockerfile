# Dockerfile

# Use the official Python image as a base

FROM python:3.10-slim

# https://katalon-inc.my.site.com/katalonhelpcenter/s/article/How-to-install-FFMPEG-on-Docker-Image
# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container

WORKDIR /app

# Copies requirements.txt from your local app folder into the container.

COPY app/requirements.txt .

# Installs FastAPI, Uvicorn, and other dependencies listed in requirements.txt.

RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code to the working directory
COPY app /app

# Exposes port 80, which is used to access the application from outside the container.
EXPOSE 80

# Starts the FastAPI app with Uvicorn, listening on all network interfaces.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]