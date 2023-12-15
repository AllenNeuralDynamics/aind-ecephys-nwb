# hash:sha256:cfb3c699705361d463e993e633591e391f82069ac60c3ea8894dc9e901daa4e4
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    neuroconv==0.4.6 \
    nwbwidgets==0.11.3 \
    pynwb==2.5.0 \
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5
