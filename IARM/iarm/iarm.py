#!/usr/bin/python3

from crypt import methods
import string
from flask import Flask, request, Response
import json
import copy
from threading import *
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from matplotlib.style import available
from numpy import int16, int8
import yaml
from yaml.loader import FullLoader
import sys
import re
from typing import Optional
from kubernetes import client, config
import time
import redis
import pickle
import requests
from typing import List
from threading import Thread, Lock



divider="****************"
config.load_kube_config(config_file='../../.kube/config')
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()

mutex=Lock()

offline_map=dict()
meo = 'http://localhost:5001'


resources_allocatable=dict()
resources_available=dict()

def convert_to_ki(memory_resource):
    unit = re.search("(Ki|Mi)$", memory_resource)
    if not unit:
        return memory_resource
    unit=unit[0]
    if unit == "Ki":
        mult=1
    elif unit == "Mi":
        mult= 1024
    else:
        print("Help me")

    return float(re.search("([0-9]*[.])?[0-9]+",memory_resource)[0]) * mult

def convert_to_cpu(cpu):
    unit = re.search("(m|millicpu)$", cpu)
    if not unit:
        return cpu
    unit = unit[0]
    # print(unit)
    if not unit:
        mult=1
    elif unit == "millicpu":
        mult=0.001
    elif unit == "m":
        mult= 0.001
    else:
        print("Help me")
    return float(re.search("([0-9]*[.])?[0-9]+",cpu)[0]) * mult

def update_resources():
    mutex.acquire()
    res_pods = v1.list_pod_for_all_namespaces()
    global resources_available
    resources_available = copy.deepcopy(resources_allocatable)
    for pod in res_pods.items:
        for container in pod.spec.containers:
            # print(pod.metadata.name, container.resources.requests, pod.spec.node_name)
            if not container.resources.requests:
                continue
            for key in container.resources.requests:
                if key == 'memory':
                    rss = convert_to_ki(container.resources.requests[key])
                elif key == 'cpu':
                    rss = convert_to_cpu(container.resources.requests[key])
                    # print(pod.metadata.name, rss, pod.spec.node_name)
                else:
                    rss = container.resources.requests[key]
                    # if "xilinx" in key:
                        # print(divider)
                        # print(rss)
                        # print(divider)
                # print(resources_available)
                resources_available[pod.spec.node_name][key] = float(resources_available[pod.spec.node_name][key]) - float(rss)
                # print(resources_available[pod.spec.node_name][key])
    mutex.release()
    return

def update_node_resources():
    mutex.acquire()
    for node in v1.list_node().items:
        # print(i.metadata.name)
        resources_allocatable[node.metadata.name] = node.status.allocatable
        # print(resources_allocatable[node.metadata.name])
        resources_allocatable[node.metadata.name]['memory'] = str(convert_to_ki(resources_allocatable[node.metadata.name]['memory']))
        resources_allocatable[node.metadata.name]['cpu'] = str(convert_to_cpu(resources_allocatable[node.metadata.name]['cpu']))
    mutex.release()
        





    

# print(resources_allocatable)
# print(divider)
update_node_resources()
resources_available = copy.deepcopy(resources_allocatable)
update_resources()
# print (resources_available)

@dataclass
class accel_entry:
        version_list: List=field(default_factory=lambda:[])
        accelerator_type: string=""
        name: string=""
        namespace: string=""
        latency_req: int = -2
        rps_req: float = -2.0
        energy_req: int=-2
        last_mon: datetime= (2020, 5, 17)
        scheduled_at: datetime=(2020, 5, 17)
        version_name: string=""
        

@dataclass
class AIF_entry:
        accel: accel_entry
        service_name: string=""
        deployment_name: string=""
        namespace: string=""
        yaml_file: string=""

r = redis.Redis(
    host='127.0.0.1',
    port=6379
)

# class my_custom_unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == " __main__":
#             module="program"
#         return super().find_class(module, name)


@dataclass
class profile:
    accel_type:string
    latency_ns: int
    energy_class: int8

# Fill/Update this map based on continuous learning?
# or from a file

# or just fill it once?

# XXX IARM needs to be acknowledged if a specific request for deployment is finally successfully served by MEC PM and VIM,
# in order to update the local key-value store of managed accelerated functions

# Dictionary: ID -> entry


# @dataclass
# class entry:
#     accelerator_type: string
#     yaml_file: string
#     name: string
#     namespace: string
#     latency_req: int = -2
#     rps_req: float = -2.0
#     energy_req: int=-2
#     last_mon: datetime= (2020, 5, 17)
#     scheduled_at: datetime=(2020, 5, 17)
#     version_name: string=""

@dataclass
class metrics:
    throughput: list
    latency: list
    avg_latency: float=0
    avg_throughput: float=0
    max_latency: float=0
    
def get_metrics_from_db():
    return ""

def check_health():
    return ""

app=Flask(__name__)



@app.route('/api/register',methods=['POST'])
def register():
    # Get metrics sync or async
    data = request.get_json()
    objectives=data['objectives']    
    print(objectives)
    
    curr_app_pickled=r.get(data['name']+":"+data['namespace'])
    curr_app=pickle.loads(curr_app_pickled)

    # -1 means best-effort
    if 'latency' in objectives:
        print("Found latency")
        latency_req = objectives['latency']
        curr_app.accel.latency_req=latency_req
    if 'throughput' in objectives:
        print("Throughput")
        rps_req = objectives['throughput']
        curr_app.accel.rps_req=rps_req
    if 'energy' in objectives:
        print("Energy")
        energy_req = objectives['energy']
        curr_app.accel.energy_req=energy_req

    
    available_versions=data['availableVersions']
    print(available_versions)
    sorted_versions=sort_versions(available_versions,curr_app)
    found=False
    i=0
    while not is_resource_available(sorted_versions[i]['device'],1):
        # resource_to_request=versions_list[i]
        print(is_resource_available(sorted_versions[i]['device'],1))
        i=i+1

    
    # resource_to_request=sorted_versions[i]
    selected_version=sorted_versions[i]
    print(i,selected_version)
    # selected_version=select_version(available_versions,curr_app)
    # resource_to_request = selected_version['device']

    curr_app.accel.last_mon=datetime.now()
    curr_app.accel.scheduled_at=datetime.now()
    curr_app.accel.version_name=selected_version['name']
    curr_app.accel.accelerator_type=selected_version['device']
    curr_app.yaml_file=selected_version['configFile']
    curr_app.accel.version_list=sorted_versions
    # print("Register: Versions List", curr_app.version_list)

    # Change the name depending on the selected version
    # curr_app.name=selected_version['name']

    # Store in the KV 
    r.set(data['name']+":"+data['namespace'],pickle.dumps(curr_app))
    # Store it remotely and i) ask sync/async the MEO whether the deployment succeeded
    # running_applications[(curr_app.name,curr_app.namespace)] = curr_app
    # metrics_of_apps[(curr_app.name,curr_app.namespace)]=metrics(latency=[],throughput=[])

    return Response(response=json.dumps(selected_version),status=200,mimetype='application.json')


@app.route('/api/alter',methods=['POST'])
def alter():
    # Get metrics sync or async
    data = request.get_json()


    # Search for the service
    # if not r.exists((data['name'], data['namespace'])):
    #     return Response(response=json.dumps("The service was not found in IARM entries"),status=200,mimetype='application.json')

    # Check for changes in objectives, available versions
    # found_entry=running_applications[(data['name'],data['namespace'])]
    found_entry_pickled= r.get(data['name']+":"+data['namespace'])
    found_entry=pickle.loads(found_entry_pickled)

    objectives=data['objectives']    
    # print(objectives)
    if 'latency' in objectives:
        print("Found latency")
        latency_req = objectives['latency']
        if latency_req == found_entry.accel.latency_req:
            print("No change in latency requirements")
        else:
            found_entry.accel.latency_req=latency_req
    else:
        found_entry.accel.latency_req=-2

    if 'throughput' in objectives:
        print("Throughput")
        rps_req = objectives['throughput']
        if rps_req == found_entry.accel.rps_req:
            print("No change in throughput requirements")
        else:
            found_entry.accel.rps_req=rps_req
    else:
        found_entry.accel.rps_req=-2
    
    if 'energy' in objectives:
        print("Energy")
        energy_req = objectives['energy']
        if energy_req == found_entry.accel.energy_req:
            print("No change in energy requirements")
        else:
            found_entry.accel.energy_req=energy_req
    else:
        found_entry.accel.energy_req=-2


    available_versions=data['availableVersions']
    print("Available versions:", available_versions)
    selected_version=select_version(available_versions,found_entry)
    print("Selected version:", selected_version)
    resource_to_request = selected_version['device']
    # Compare the current with the requested versions
    if selected_version['name'] == found_entry.accel.version:
        # We have the same version
        # Let's check the yaml file as well to be sure
        if selected_version['configFile'] == found_entry.yaml_file:
            # We have the same config
            # do nothing
            # exit
            status=400
            ret="No different version was chosen to be deployed"
            
        else:
            status=400
            ret="Same version name different config file"
    else:
        if selected_version['configFile'] == found_entry.yaml_file:
            status=400
            ret="Different version name, same config file"
        else:
            # ret="Different version name, different config file"
            found_entry.accel.scheduled_at=datetime.now()
            found_entry.accel.version=selected_version['name']
            found_entry.accel.accelerator_type=selected_version['device']
            found_entry.yaml_file=selected_version['configFile']
            r.set(data['name']+":"+data['namespace'],pickle.dumps(found_entry))
            # running_applications[(found_entry.name,found_entry.namespace)] = found_entry
            ret=selected_version
            status=200

    return Response(response=json.dumps(ret),status=status,mimetype='application.json')

def get_metric(version, metric):
    val = version['optimization'][metric]
    if metric == 'latency':
        val = convert_to_ms(version['optimization'][metric])
    return val

def sort_versions(versions, app):
    # reverse=False
    version_list=[]
    if app.accel.latency_req==-1:
        sorted_versions=sorted(versions,key=lambda x:convert_to_ms(x['optimization']['latency']), reverse=False)
    else:
        sorted_versions=sorted(versions,key=lambda x:x['optimization']['throughput'], reverse=True)
    
    print("Versions not sorted:",versions)
    print("Requiremenents:", app.accel.latency_req)
    print("Versions sorted:",sorted_versions)
    # for version in versions:
    #     if isValid(version['optimization'][metric]):
    #         if metric=='latency':
    #             curr = convert_to_ms(version['optimization'][metric])
    #         else:
    #             curr=version['optimization'][metric]
    #         version_list.append(version)

    # return version_list.sort(reverse=reverse, key=get_metric)
    return sorted_versions    

# This is limited to the best effort choices (latency/throuhput)
def select_version(versions,app):
    if app.accel.latency_req == -1:
            # select the latency optimization version
            # or version with minimum latency
            min_lat=sys.maxsize * 2 + 1
            for version in versions:
                if is_resource_available(version['device'],1):
                    if 'latency' in version['optimization']:
                        if isValid(version['optimization']['latency']):
                                curr_lat = convert_to_ms(version['optimization']['latency'])
                        if curr_lat < min_lat:
                            min_lat = version['optimization']['latency']
                            selected_version = version
            
        # If we need best-effort for throughput
    elif app.accel.rps_req == -1:
        max_rps=0
        for version in versions:
            if 'throughput' in version['optimization']:
                if version['optimization']['throughput'] > max_rps:
                    max_rps = version['optimization']['throughput']
                    selected_version = version
    else:
        # TODO
        selected_version=""
        print("Neither latenct=best-effort nor throughput=best-effort")    

    return selected_version

def isValid(latency):
    # return re.search("^[0-9]*(s|ms)$", latency)
    return re.search("([0-9]*[.])?[0-9]+(s|ms)$",latency)

def convert_to_ms(input):
    unit = re.search("(s|ms|h)$", input)[0]
    if unit == "s":
        mult=1000.0
    elif unit == "m":
        mult= 60000.0
    else:
        mult=1.0

    return float(re.search("([0-9]*[.])?[0-9]+",input)[0]) * mult


# # One thread for continuously monitoring devices
# y = Thread(target=get_metrics_from_db, daemon=True)
# y.start()

# # One thread for checking health
# x = Thread(target=check_health(), daemon=True)
# x.start()


# print("This is main thread")

def add_logs(name,ns,logs):
    logs_per_line=logs.split("\n")
    print("logs per line:",logs_per_line)
    for log in logs_per_line:
        print("log:",log)
        if 'Execution time' in log:
            print("Found execution time")
            lat= log.split(":")[3]
            print(lat)
            if isValid(lat):
                print("is valid")
                lat2=convert_to_ms(lat)
                print(lat2)
                metrics_of_apps[(name,ns)].latency.append(lat2)
    print(metrics_of_apps)
    return

def metrics_collector():
    while True:
        print("Alert! It's time to update the logs")
        for (name,ns) in running_applications:
            # Find the app (combination of name/namespace)
            # Usually app==deployment; thus check all the available pods of this deployment
            # get their logs
            ret = v1.list_namespaced_pod(namespace=ns,field_selector='status.phase==Running')
            for pod in ret.items:
                pod_name=pod.metadata.name
                print(pod_name)
                if name in pod_name:
                    logs = v1.read_namespaced_pod_log(pod_name,
                        namespace=ns,since_seconds=5)
                    print(logs)
                    add_logs(name,ns,logs)


        # for node in get_nodes():
        #     print("[state_collector]#",node)
        #     update_local_kv(node)
        # print("Finished updating the local_kv entries")
        time.sleep(5)

# def is_resource_available(resource_name):
#     return get_node_accel_resources(resource_name) > 0

def get_node_accel_resources(resource_name):
    nodes = v1.list_node()
    accel_resource_counter=0
    for i in nodes.items:
        print(i.metadata.name)
        if resource_name in i.status.allocatable:
            accel_resource_counter = accel_resource_counter+ int(i.status.allocatable[resource_name])
        # print(i.status.allocatable)
        # for rss in i.status.allocatable:
        #     if resource_name in rss:
        #         # print(i.status.allocatable[rss])
        #         accel_resource_counter = accel_resource_counter+1
    return accel_resource_counter

def is_resource_available(resource_name, q):
    nodes = v1.list_node()
    for node in nodes.items:
        # print(divider)
        # print(node.metadata.name)
        # print(divider)
        # print(divider)
        if resource_name not in resources_available[node.metadata.name]:
            continue
        if float(resources_available[node.metadata.name][resource_name]) >= q:
            return True
    return False



def check_for_improvement():
    suffix="/api/system/migrate"
    # store hardware availability in a temp dict --> this can be done dynamically
    # For each running app
        # check the sorted versions
        # if a version with higher priority is found -> migrate
    time.sleep(10)
    while True:
        time.sleep(5)
        # update_node_resources()
        update_resources()
        print("Capacity:", resources_allocatable)
        print ("Availability:",resources_available)
        for key in r.scan_iter("*"):
            # print(key)
            # print(type(key))
            entry_pickled=r.get(key)
            entry=pickle.loads(entry_pickled)
            print(key)
            if not entry.accel:
                # This AIF is not managed by IARM
                print(key, ": is not an AI accel func")
                continue
            i=0
            # print("Version List:", entry.accel.version_list)
            # print("Current version name", entry.accel.version_name)
            while i < len(entry.accel.version_list) and entry.accel.version_list[i]['name'] != entry.accel.version_name:
                if is_resource_available(entry.accel.version_list[i]['device'], 1):
                    print("FOUND RESOURCE AVAILABLE")
                    print(i)
                    # data=
                    # There are 2 Options:
                    # 1. Send old and new yaml files
                    # 2. Send only new yaml file and delete using the app label
                    key2=key.decode()
                    data={
                        "name": key2.split(':')[0],
                        "namespace":key2.split(':')[1],
                        "oldyaml": entry.yaml_file,
                        "newyaml": entry.accel.version_list[i]['configFile']
                    }
                    response = requests.post(meo+suffix, json=data)
                    print(response)
                    print("Migrating", entry.deployment_name, "to --->",entry.accel.version_list[i]['device'])
                    # Change the accel yaml file
                    # Change the entries of accel:
                        # yaml file
                        # device
                    entry.accel.version= entry.accel.version_list[i]['name']
                    entry.accel.accelerator_type= entry.accel.version_list[i]['device']
                    break
                i=i+1
        

        

# This thread will monitor all running apps and print metrics, e.g. latency, throughput
# y = Thread(target=metrics_collector, daemon=True)
# y.start()

# This thread will poll the available resources in order to migrate the application if possible
y = Thread(target=check_for_improvement, daemon=True)
y.start()


if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5002)