docker run -it --rm --name sc3_gpu --gpus '"device=1"' --network=host aimilefth/ai_at_edge_demo:sc3_gpu

docker exec -ti sc3_gpu bash

port:3001
