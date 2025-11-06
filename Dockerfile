FROM python:3.10-slim

WORKDIR /app
COPY . .

# 🧩 Add required system libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 🧩 Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 🧩 Use gunicorn (more stable than Flask dev server)
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
