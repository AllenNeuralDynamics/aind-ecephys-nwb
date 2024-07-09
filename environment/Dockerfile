# hash:sha256:1d0c6284daec69f41dfd301a23bc5026b9f23ef26e2bf59cd3d02e9928421209
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.7.0 \
    neuroconv==0.4.9 \
    pynwb==2.7.0 \
    spikeinterface[full,widgets]==0.100.8 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
