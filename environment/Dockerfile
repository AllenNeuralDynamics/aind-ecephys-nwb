# hash:sha256:f8618a28d36e66d0477ff9b9a56653eb07d78705e85fd9c777886b4382665b30
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
    spikeinterface[full]==0.101.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
