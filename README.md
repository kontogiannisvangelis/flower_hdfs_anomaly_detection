```
DOCKER_BUILDKIT=1 docker build -t flower_hdfs .
```

```
docker run --gpus all -it \
  -v $(pwd)/results:/app/results \
  -v /home/ekontogi/Diploma-Thesis/hdfs_anomaly_detection/data/tokenized_dataset:/data \
  flower_hdfs \
  flwr run .
```
