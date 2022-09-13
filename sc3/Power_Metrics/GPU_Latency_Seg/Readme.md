docker run -it --rm --name sc3_gpu_power --gpus '"device=1"' aimilefth/ai_at_edge_demo:sc3_gpu_power

docker exec -ti sc3_gpu_power bash

docker cp sc3_gpu_power:/home/Documents/logs/<log_file_name> .
