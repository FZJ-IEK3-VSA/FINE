#!/bin/bash
envname=DESIRED_ENVIRONMENT_NAME


conda install mamba -c conda-forge
mamba create -n $envname
source activate $envname

mamba env update -n $envname --file=requirements-dev.yml
mamba info --envs
source activate $envname
cd ..
git clone --branch dev https://jugit.fz-juelich.de/iek-3/shared-code/geokit.git
cd geokit
pip install -e .
python -m pip show geokit
cd ..
git clone --branch develop https://github.com/FZJ-IEK3-VSA/FINE
cd FINE
pip install -e .
cd ..
cd modelBuilder
python -m pip install -e .

