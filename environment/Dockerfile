# hash:sha256:392a5126fc7e029746050805c47929f3bb90cd0c5d803bdcd743dd5013dbaf75
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
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
    aind-nwb-utils==0.1.0 \
    hdmf-zarr==0.11.0 \
    neuroconv==0.7.1 \
    pynwb==2.8.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.103.0 \
    wavpack-numcodecs==0.2.2
