# Usar una imagen base ligera con soporte para CUDA (si lo necesitas)
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Evitar preguntas interactivas en la instalación de paquetes
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar el sistema e instalar dependencias esenciales
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libgl1-mesa-glx \
    libjpeg62 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Configurar Conda en el PATH
ENV PATH=/opt/conda/bin:$PATH

# Inicializar Conda
RUN /opt/conda/bin/conda init bash

# Crear carpetas necesarias para el proyecto
WORKDIR /app
RUN mkdir -p /app/input /app/output /app/tmp

# Copiar los contenidos actuales al contenedor
ADD . /app

# Crear el entorno Conda y activar
RUN /bin/bash -c "source ~/.bashrc && conda activate base && conda create -n v2e python=3.10 -y"

# Instalar dependencias desde un archivo requirements.txt
COPY requirements.txt /app/requirements.txt
RUN /bin/bash -c "source ~/.bashrc && conda activate v2e && pip install -r /app/requirements.txt"

# Comando por defecto
CMD ["/bin/bash"]
