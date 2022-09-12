#!/bin/bash

IMAGE_NAME=$1
COMMAND=$2

#uid=`id -u`
#gid=`id -g`

DETACHED="-it"

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done


docker_run_params=$(cat <<-END
    -v /dev/shm:/dev/shm \
    -v /opt/xilinx/dsa:/opt/xilinx/dsa \
    -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -v /etc/xbutler:/etc/xbutler \
    -e VERSION=$VERSION \
    --rm \
    --network=host \
    --name=sc3_alveo \
    ${DETACHED} \
    $IMAGE_NAME \
    $COMMAND
END
)

docker run \
  $docker_devices \
  $docker_run_params


