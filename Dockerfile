FROM continuumio/miniconda3:latest

LABEL maintainer="Julian Schönau"

EXPOSE 8888

RUN apt-get -y update
RUN apt-get -y upgrade

COPY . ./fine
WORKDIR "./fine"

RUN conda install mamba -c conda-forge
RUN mamba env update -n fine --file=requirements.yml
RUN mamba env update -n fine --file=requirements_dev.yml
RUN /opt/conda/envs/fine/bin/pip install -e .
RUN mamba install -n fine -c conda-forge coincbc

RUN echo "source activate fine" > ~/.bashrc
ENV PATH /opt/conda/envs/fine/bin:$PATH

CMD ["jupyter", "notebook", "--notebook-dir=./examples", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]