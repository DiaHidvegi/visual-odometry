# 1. Download, install & set up Docker Desktop (https://www.docker.com/products/docker-desktop/)
#    Optional:
#     - Start Docker Desktop (should automatically start Docker Engine)
#     - Run 'docker pull continuumio/miniconda3:24.11.1-0' in cmd to install base image
# 2. Go to base directory that contains the Dockerfile
# 3. run 'docker build -t vamr_proj .' to build container
# 4. run 'docker run -it --rm vamr_proj' to run the code

# TODO: This successfully runs the project but is unable to show the plots

FROM continuumio/miniconda3:24.11.1-0

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and create environment
COPY vamr_proj.yaml .
RUN conda env create -f vamr_proj.yaml
RUN echo "conda activate vamr_proj" >> ~/.bashrc
ENV PATH=/opt/conda/envs/vamr_proj/bin:$PATH

# Copy application code
COPY . .

# Run the application
CMD ["python", "main.py"]
