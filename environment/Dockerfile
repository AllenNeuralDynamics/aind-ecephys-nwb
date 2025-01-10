# hash:sha256:006652871f9344a54b92a9c5485c3fcc7ebecea32a57e25c95e708c249c30605
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-log-utils==0.2.3 \
    hdmf-zarr==0.8.0 \
    neuroconv==0.6.0 \
    pynwb==2.8.1 \
    spikeinterface[full]==0.101.2 \
    wavpack-numcodecs==0.2.1

COPY postInstall /
RUN /postInstall
