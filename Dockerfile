FROM mambaorg/micromamba:latest

LABEL authors="Kevin Knosala,Julian Schönau"

# Port for jupyter notebook
EXPOSE 8888

COPY --chown=$MAMBA_USER:$MAMBA_USER . ./fine
WORKDIR "./fine"

RUN micromamba install -y -n base -f requirements_dev.yml && \
    micromamba clean --all --yes

CMD ["jupyter", "notebook", "--notebook-dir=./examples", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]