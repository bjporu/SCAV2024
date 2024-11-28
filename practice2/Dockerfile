# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*



# Install Python dependencies for main.py to work
RUN pip install --no-cache-dir fastapi uvicorn numpy scipy pydantic

# Expose port 8000 to the outside world :))
EXPOSE 8000

# Define the command to run your app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
