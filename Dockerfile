##############################################
# Stage 1 — Builder: install dependencies
##############################################
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel AS builder

# Install basic system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a temporary prefix directory
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --prefix=/tmp/venv -r /tmp/requirements.txt && \
    pip install --prefix=/tmp/venv jupyterlab numpy opencv-python


##############################################
# Stage 2 — Runtime
##############################################
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Lightweight runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy Python site-packages from builder
COPY --from=builder /tmp/venv /usr/local

# Allow python to find installed packages
ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:${PYTHONPATH}"


##############################################
# Stage 3 — Code + Model Weights
##############################################
WORKDIR /code

# Copy the full source code
COPY . /code

# Copy model weights inside the image
COPY ./saved_models /code/saved_models

# Ensure output folder exists
RUN mkdir -p /result

# Make scripts executable
RUN chmod +x /code/predict.sh
RUN chmod +x /code/start_jupyter.sh

# Default: run predict.sh
CMD ["bash", "/code/predict.sh"]
