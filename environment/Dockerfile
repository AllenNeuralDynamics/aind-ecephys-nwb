# hash:sha256:395b611eed31858b2d7db426ce9644c5c1f7d86f6f678252afa8d0c910ff169e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.6.0 \
    neuroconv==0.4.9 \
    nwbwidgets==0.11.3 \
    pynwb==2.6.0 \
    spikeinterface[full,widgets]==0.100.8 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
