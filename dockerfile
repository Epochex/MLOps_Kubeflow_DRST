FROM python:3.11-slim as base
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt && \
    pip install boto3 && \
    pip install kfp==2.0.1
ENV PYTHONPATH /app
# ----- producer
FROM base as producer
ENTRYPOINT ["python","kafka_streaming/producer.py"]
# ----- offline
FROM base as offline
ENTRYPOINT ["python","-m","ml.train_offline"]
# ----- consumer
FROM base as consumer
ENTRYPOINT ["python","kafka_streaming/consumer.py"]
