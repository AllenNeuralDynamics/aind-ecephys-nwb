# hash:sha256:908f8b029f609788e56c15e658712eac2e8f83c8fe6675b19c866311358fa997
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.4.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    aind-nwb-utils==0.1.1 \
    hdmf==4.3.1 \
    hdmf-zarr==0.12.0 \
    neuroconv==0.9.3 \
    pynwb==3.1.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.104.2 \
    wavpack-numcodecs==0.2.2
