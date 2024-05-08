FROM python:3.10

#Set the working directory in the container
WORKDIR /work


COPY src /work/src
COPY data /work/data
COPY notebooks /work/notebooks

COPY ./docker/jupyter/requirements.txt /work/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt


RUN pip install jupyter==1.0.0 -U && pip install jupyterlab==4.1.6

# Install Jupyter and JupyterLab

# Make port 8888 available to the world outside this container
EXPOSE 8888

ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY}
ENV AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_ACCESS_KEY}
# RUN mkdir /work
# COPY . .
# RUN pip install jupyter -U && pip install jupyterlab
# RUN pip install -r requirements.txt

