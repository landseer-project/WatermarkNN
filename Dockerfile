FROM continuumio/miniconda3

WORKDIR /app

RUN conda create -n watermarknn python=3.6 -y
SHELL ["conda", "run", "-n", "watermarknn", "/bin/bash", "-c"]

RUN conda install pytorch=0.4.1 torchvision=0.2.1 cpuonly -c pytorch -y

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/adiyoss/WatermarkNN.git /app

ENV COLUMNS=80

ADD docker_run.sh /app/.
CMD ["bash"]
#CMD ["conda", "run", "-n", "watermarknn", "python", "train.py", "--batch_size", "100", "--max_epochs", "60", "--runname", "train", "--wm_batch_size", "2", "--wmtrain"]

