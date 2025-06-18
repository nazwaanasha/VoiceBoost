# Menggunakan base image Python yang sesuai
FROM python:3.8-slim

# Install FFmpeg dan dependensi lainnya
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Set Working Directory
WORKDIR /app

# Copy requirements.txt dan install dependensi Python
COPY requirements.txt /app/
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variable untuk Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PATH="/opt/venv/bin:$PATH"

# Copy seluruh kode aplikasi ke dalam Docker container
COPY . /app/

# Menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "app.py"]