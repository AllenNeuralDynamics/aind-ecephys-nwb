# hash:sha256:bcdf574d6585bfd8547a4a0f15c04ef701b03c36829316aeafba2ffbd77ef5b4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

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
    hdmf-zarr==0.11.0 \
    neuroconv==0.7.1 \
    pynwb==2.8.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.0 \
    wavpack-numcodecs==0.2.2
