# hash:sha256:5f154ba9728b5763e9f73456d8638affc57cc9af5d75c0728795f474c448ca30
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    aind-nwb-utils==0.1.1 \
    hdmf==4.3.1 \
    hdmf-zarr==0.13.0 \
    neuroconv==0.9.3 \
    pynwb==3.1.3 \
    s3fs==2025.5.1 \
    spikeinterface[full]==0.104.7 \
    wavpack-numcodecs==0.2.2
