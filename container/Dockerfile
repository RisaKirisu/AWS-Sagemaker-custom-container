# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flas`k stack
# for serving inferences in a stable way.

# Local test:   docker build -t test_exllama .
#               docker run --rm --name Llama-70b --gpus all -p 8080:8080 -v <test_dir>:/opt/ml test_exllama serve

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Here we get all python packages.
RUN pip3 --no-cache-dir install flask gunicorn

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
ADD exllamav2-0.0.6-cp310-cp310-linux_x86_64.whl /
RUN pip3 install exllamav2-0.0.6-cp310-cp310-linux_x86_64.whl

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY llama_2 /opt/program
WORKDIR /opt/program