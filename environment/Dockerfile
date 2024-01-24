# hash:sha256:2c2f3323d17958bf3e162757df7df2f37079bf93b951dff9a123aa927fc6c4e4
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    hdmf==3.12.0 \
    hdmf-zarr==0.5.0 \
    neuroconv==0.4.6 \
    nwbwidgets==0.11.3 \
    pynwb==2.5.0 \
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
