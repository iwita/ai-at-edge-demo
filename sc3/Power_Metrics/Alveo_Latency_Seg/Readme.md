#docker run -it --rm --name sc3_alveo --network=host aimilefth/ai_at_edge_demo:sc3_alveo
./docker_run.sh aimilefth/ai_at_edge_demo:sc3_alveo

docker exec -ti sc3_alveo bash

port 3000

Need to run source setup.sh U280_L from inside the container
