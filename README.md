## Docker Deployment
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

## Copy to server
```
rsync -avz \
    --exclude-from='./flower_hdfs_anomaly_detection/.dockerignore' \
    -e "ssh -i ~/.ssh/key" \
    ./flower_hdfs_anomaly_detection vkontogiannis@gpu.epu.ntua.gr:/mnt/datadrive/vkontogiannis \
    --dry-run
```