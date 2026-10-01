# hash:sha256:4686c662b966753298020711690a38c515a0b31833c6f8ddfa2e4520727112cc
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
        jupyterlab==4.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-nwb-utils==0.2.9 \
    hdmf==6.1.0 \
    hdmf-zarr==0.13.0 \
    log-schema==0.2.10.dev1 \
    neuroconv==0.9.3 \
    pynwb==4.0.0 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.105.0 \
    wavpack-numcodecs==0.2.2
