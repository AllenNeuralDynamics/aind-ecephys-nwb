# hash:sha256:0b9b727141170b3f9d666e13300f84ff3ee2b6ecae460affa7aeb7ea8e76ea82
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
    spikeinterface[full]==0.101.2 \
    wavpack-numcodecs==0.2.1

COPY postInstall /
RUN /postInstall
