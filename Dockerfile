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

# Install Python dependencies into temp prefix
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --prefix=/tmp/venv -r /tmp/requirements.txt && \
    pip install --prefix=/tmp/venv jupyterlab numpy opencv-python


##############################################
# Stage 2 — Runtime (clean)
##############################################
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Install required runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /tmp/venv /usr/local

# Make sure site-packages are visible
ENV PYTHONPATH="/usr/local/lib/python3.8/site-packages:${PYTHONPATH}"


##############################################
# Stage 3 — Add Code + Model Weights
##############################################
WORKDIR /code

# Copy your source code
COPY . /code

# This ensures predict.py can load them inside Docker.
COPY ./saved_models /code/saved_models

# Ensure result directory exists
RUN mkdir -p /result

CMD ["bash", "/code/predict.sh"]
