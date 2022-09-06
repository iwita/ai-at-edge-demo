docker run -it --rm --name sc2_gpu --gpus '"device=1"' --network=host aimilefth/ai_at_edge_demo:sc2_gpu

docker exec -ti sc2_gpu bash

port:3001
