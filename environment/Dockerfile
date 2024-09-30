# hash:sha256:af95bbc822b50085514792eb026f1bcaf42f08d25769018b47224d265a3e4689
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf-zarr==0.8.0 \
    neuroconv==0.6.0 \
    pynwb==2.8.1 \
    spikeinterface[full]==0.101.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
