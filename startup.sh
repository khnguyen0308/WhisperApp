#!/bin/bash

# Step 1: Log and update package lists
echo "--- Startup Script: Updating package list ---"
apt-get update -y

# Step 2: Log and install ffmpeg
echo "--- Startup Script: Installing ffmpeg ---"
apt-get install -y ffmpeg

# Step 3: Log and start Gunicorn server
echo "--- Startup Script: Starting Gunicorn ---"
gunicorn --bind=0.0.0.0:8000 --timeout 600 app:app