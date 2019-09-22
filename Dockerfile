FROM continuumio/miniconda:latest

ENV ENV_NAME=env
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

RUN conda update conda && \
    conda create -n ${ENV_NAME} python=3.6 && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    conda info --envs && \
    pip install numpy pandas matplotlib==3.1.0 seaborn scikit-learn mlflow && \
    conda install -c conda-forge lightgbm xgboost && \
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc
