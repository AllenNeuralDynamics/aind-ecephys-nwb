# hash:sha256:c3ccca7f7b000848ed55c606cae81c19e5e5f049b1149ba8fe7753343d87e4a7
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.8.0 \
    neuroconv==0.4.9 \
    pynwb==2.8.1 \
    spikeinterface[full]==0.101.0 \
    wavpack-numcodecs==0.1.5
