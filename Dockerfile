FROM python:3.10-slim

# Install system libraries needed for MediaPipe + OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Use gunicorn in production, not python app.py
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
