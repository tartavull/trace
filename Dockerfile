FROM tensorflow/tensorflow:nightly-devel

MAINTAINER Ben Eisner <beisner@princeton.edu>

# Pick up some Neuroglancer dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libtiff5-dev \
        libhdf5-dev \
        libjpeg8-dev \
        zlib1g-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libgmp-dev \
        libwebp-dev \
        tcl8.6-dev \
        tk8.6-dev \
        python-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Julia
RUN add-apt-repository ppa:staticfloat/juliareleases && \
        add-apt-repository ppa:staticfloat/julia-deps && \
        apt-get update && \
        apt-get install -y julia

# Add some pip dependencies
RUN pip --no-cache-dir install \
        cython \
        requests \
        tqdm

# Get the code for trace
RUN git clone --depth=50 https://github.com/tartavull/trace.git /home/trace/
WORKDIR /home/trace
RUN git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    make submodules

# Make sure we can get all remote branches
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && \
    git fetch


# Download the h5 images
WORKDIR /home/trace/trace
RUN python cli.py download

# Neuroglancer
EXPOSE 4125
