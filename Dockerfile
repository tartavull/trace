FROM tensorflow/tensorflow

ARG BRANCH=master

MAINTAINER Ben Eisner <beisner@princeton.edu>

# Pick up some Neuroglancer dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libtiff5-dev \
        libhdf5-dev \
        libjpeg8-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libgmp-dev \
        libwebp-dev \
        libffi-dev \
        openssl \
        tcl8.6-dev \
        tk8.6-dev \
        python-tk \
        build-essential \
        libssl-dev \
        libffi-dev \
        python-dev \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Julia
RUN add-apt-repository ppa:staticfloat/juliareleases && \
        add-apt-repository ppa:staticfloat/julia-deps && \
        apt-get update && \
        apt-get install -y julia

# Install some convenience tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        screen \
        tmux \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add some pip dependencies
RUN pip --no-cache-dir install \
        cython \
        requests \
        tqdm \
        munkres \
        scikit-image \
        opencv-python

# Install the cremi dataworks
RUN pip install git+https://github.com/cremi/cremi_python.git

# Get the code for trace
RUN git clone --depth=50 https://github.com/tartavull/trace.git /home/trace/
WORKDIR /home/trace
RUN git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    pip install -r requirements_dev.txt && \
    make submodules

# Make sure we can get all remote branches
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git fetch

# Checkout the specified branch
RUN git checkout $BRANCH

# Download the h5 images
WORKDIR /home/trace/trace
RUN mkdir isbi/
RUN mkdir snemi3d/
RUN mkdir -p cremi/a/
RUN mkdir -p cremi/b/
RUN mkdir -p cremi/c/
RUN python cli.py download

# Neuroglancer
EXPOSE 4125
