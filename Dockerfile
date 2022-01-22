FROM tensorflow/tensorflow:2.6.1-gpu-jupyter


# install packages (incl. opencv)
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt