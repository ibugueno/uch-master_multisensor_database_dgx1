# Usar una imagen base ligera de Ubuntu 20.04
FROM ubuntu:20.04

# Establecer variables de entorno para evitar interacciones durante la construcción
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar paquetes e instalar dependencias necesarias
RUN apt-get update && \
    apt-get install -y zip wget libgl1 libglib2.0-0 vim tmux build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Configurar Conda en el PATH
ENV PATH=/opt/conda/bin:$PATH

# Crear el environment base e instalar Python
RUN /opt/conda/bin/conda create -n base python=3.8 -y && \
    echo "conda activate base" > ~/.bashrc

# Activar Conda y configurar el shell
SHELL ["/bin/bash", "--login", "-c"]

# Instalar YOLO y dependencias
RUN conda activate base && \
    pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install opencv-python

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
RUN mkdir /app/input
RUN mkdir /app/output
RUN mkdir /app/tmp

# Configurar el entorno por defecto en bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc


# Comando por defecto (puedes cambiarlo si es necesario)
CMD ["bash"]
