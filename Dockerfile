# Base image with Python 3.10
FROM python:3.10-slim

# Install system dependencies (including OpenGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    tesseract-ocr \
    poppler-utils && \
    apt-get clean



# Set work directory
WORKDIR /app

# Set environment variables to enable Surya model compilation
ENV COMPILE_RECOGNITION=true \
    COMPILE_DETECTOR=true \
    COMPILE_LAYOUT=true \
    COMPILE_TABLE_REC=true

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Expose Jupyter notebook port
EXPOSE 8888

# Install Jupyter Notebook
RUN pip install notebook

# Entry point for Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]