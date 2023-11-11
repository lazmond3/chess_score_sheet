container_name='jupyter-chess_score_sheet'
docker run \
  --gpus all \
  --rm \
  -v $(pwd):$(pwd) \
  -p 10001:8888 \
  --name ${container_name} \
  --entrypoint jupyter \
  -d \
  tensorflow/tensorflow:latest-gpu-jupyter \
  notebook --notebook-dir=$(pwd) --ip 0.0.0.0 --no-browser --allow-root

# To check access token
docker logs ${container_name}
