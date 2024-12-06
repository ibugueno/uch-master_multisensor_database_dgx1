# Usar una imagen base ligera de Ubuntu 20.04
FROM ubuntu:20.04

# Establecer variables de entorno para evitar interacciones durante la construcción
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar paquetes e instalar dependencias necesarias
RUN apt-get update && \
    apt-get install -y \
    zip wget libgl1 libglib2.0-0 vim tmux build-essential \
    python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Configurar Conda en el PATH
ENV PATH=/opt/conda/bin:$PATH

# Instalar Python y dependencias en el environment base
RUN conda install -n base python=3.8 -y && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
        tqdm \
        pillow \
        pyyaml \
        numpy \
        matplotlib \
        opencv-python \
        scikit-learn

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el contenido del proyecto
ADD . /app

# Crear carpetas para datos de entrada, salida y temporales
RUN mkdir -p /app/input /app/output /app/tmp

# Configurar el entorno Conda en bash por defecto
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Comando por defecto
CMD ["bash"]
