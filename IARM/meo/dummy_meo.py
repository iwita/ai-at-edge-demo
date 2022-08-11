#!/usr/bin/python3

from argparse import Namespace
from crypt import methods
from mimetypes import suffix_map
import string
from sys import prefix
from unicodedata import name
from flask import Flask, request, Response
import json
from threading import *
from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timedelta
from numpy import int16, int8
import yaml
from yaml.loader import FullLoader
import requests
from kubernetes import client, config
import redis
import pickle

r = redis.Redis(
    host='127.0.0.1',
    port=6379
)

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

# class accel_entry:
#         def __init__(self, name="", namespace="", accelerator_type="", latency_req=-2,
#                 energy_req=-2, rps_req=-2.0, version_name="",scheduled_at= (2020, 5, 17),
#                 last_mon= (2020, 5, 17)):
#                 self.name=name
#                 self.latency_req=latency_req
#                 self.rps_req=rps_req
#                 self.energy_req=energy_req
#                 self.version_name=version_name
#                 self.last_mon=last_mon
#                 self.scheduled_at=scheduled_at
#                 self.namespace=namespace 
#                 self.accelerator_type=accelerator_type

@dataclass
class AIF_entry:
        accel: accel_entry
        service_name: string=""
        deployment_name: string=""
        namespace: string=""
        yaml_file: string=""

# class AIF_entry:
#         def __init__(self, accel=accel_entry(), service_name="", 
#                 deployment_name="", namespace="", yaml_file=""):
#                 self.accel=accel
#                 self.service_name=service_name
#                 self.deployment_name=deployment_name
#                 self.namespace=namespace
#                 self.yaml_file=yaml_file


def check_redis():
       return r.ping() 

iarm = 'http://localhost:5002'
# test_url = iarm + '/api/schedule'

app=Flask(__name__)

config.load_kube_config(config_file='../../.kube/config')
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()

applications=dict()
service_name=dict()

# @app.route('/api/exists',methods=['POST'])
# def exists():
#         data=request.get_json()
#         print(data)
#         if (data['name'],data['namespace']) in r.get():
#                 ret="Found"
#         else:
#                 ret="Not Found"

#         return Response(response=json.dumps(ret),status=200,mimetype='application.json')

@app.route('/api/delete',methods=["POST"])
def delete():
        data=request.get_json()
        print(data)
        if not r.exists(data['name']+":"+data['namespace']):
                return Response(response=json.dumps("AIF was not found"),status=400,mimetype='application.json')
        curr_app=pickle.loads(r.get(data['name']+":"+data['namespace']))
        try:
                print(curr_app.deployment_name, curr_app.namespace)
                apps_v1.delete_namespaced_deployment(name=curr_app.deployment_name, 
                        namespace=curr_app.namespace)
        except Exception:
                print("Deployment not found")
        try:
                print(curr_app.deployment_name, curr_app.namespace)
                v1.delete_namespaced_service(name=curr_app.service_name, namespace=curr_app.namespace)
        except Exception:
                print("Service not found")
        
        # Delete entry from redis
        r.delete(data['name']+":"+data['namespace'])
        return Response(response=json.dumps("Successfully deleted "+data['name']),status=200,mimetype='application.json')

@app.route('/api/register',methods=['POST'])
def register():
        suffix = '/api/register'
        data=request.get_json()
        print(data)
        namespace=data['namespace']
        if r.exists(data['name']+":"+data['namespace']):
                return Response(response=json.dumps("AIF exists"),status=400,mimetype='application.json')


        pickled_obj = pickle.dumps(AIF_entry(namespace=data['namespace'],accel=accel_entry()))

        r.set(data['name']+":"+data['namespace'], pickled_obj)

        if data['spec']['supportsAcceleration'] == True:
                # Call the IARM
                print("Acceleration is enabled")
                response = requests.post(iarm+suffix, json=data)
                print("Response:",response.json())
        else:
                print("Acceleration is not enabled")

        # Check response

        # Apply the choice of the IARM

        # Check if service already exists: in case of migration
        response_data=response.json()
        yaml_file=response_data['configFile']
        print(yaml_file)
        # svc_yaml = response_data['configFiles']['service']
        # deployment_yaml = response_data['configFiles']['service']

        with open(yaml_file) as f:
                datas_yaml = list(yaml.load_all(f, Loader=yaml.FullLoader))
        print(datas_yaml)
        try:
                v1.create_namespaced_service(namespace=namespace,body=datas_yaml[0])
        except Exception:
                print("Service already exists")
        # Create the deployment
        apps_v1.create_namespaced_deployment(namespace=namespace,body=datas_yaml[1])

        entry=pickle.loads(r.get(data['name']+":"+data['namespace']))
        entry.deployment_name=datas_yaml[1]['metadata']['name']
        entry.service_name=datas_yaml[0]['metadata']['name']
        
        r.set(data['name']+":"+data['namespace'],pickle.dumps(entry))

        print("Entry:", entry)
        print("You can access the service via",datas_yaml[0]['spec']['ports'][0]['nodePort'])
        # Should provide the node as well

        #TODO wait to check where is it scheduler and advertise???


        # Return the response to client
        ret="Final response"
        return Response(response=json.dumps(ret),status=200,mimetype='application.json')

@app.route('/api/alter',methods=['POST'])
def alter():
        suffix = '/api/alter'
        data=request.get_json()
        print(data)
        namespace=data['namespace']
        if not r.exists(data['name']+":"+data['namespace']):
                return Response(response=json.dumps("AIF does not exist"),status=400,mimetype='application.json')
        if data['spec']['supportsAcceleration'] == True:
                # Call the IARM
                print("Acceleration is enabled")
                response = requests.post(iarm+suffix, json=data)
                print("Response:",response.json())
        else:
                print("Acceleration is not enabled")

        # Check if service already exists: in case of migration

        if response.status_code != 200:
                return Response(response=response.json(),status=response.status_code,
                        mimetype='application.json')
        response_data=response.json()
        yaml_file=response_data['configFile']
        print(yaml_file)
        # svc_yaml = response_data['configFiles']['service']
        # deployment_yaml = response_data['configFiles']['service']

        with open(yaml_file) as f:
                datas_yaml = list(yaml.load_all(f, Loader=yaml.FullLoader))
        print(datas_yaml)
        try:
                v1.create_namespaced_service(namespace=namespace,body=datas_yaml[0])
                curr_entry=r.get(data['name']+":"+data['namespace'])
                curr_entry.service_name=datas_yaml[0]['metadata']['name']
        except Exception:
                print("Service already exists")
        # Create the deployment
        apps_v1.create_namespaced_deployment(namespace=namespace,body=datas_yaml[1])
        print("The service YAML", datas_yaml[0])
        print("You can access the service via",datas_yaml[0]['spec']['ports'][0]['nodePort'])
        # Should provide the node as well

        #TODO wait to check where is it scheduler and advertise???


        # Remove the previous deployment
        # Get label
        label=datas_yaml[1]['metadata']['labels']['app']
        label_name="app="+label
        # Search deployments in the current namespace with the same tag?

        deployments=apps_v1.list_namespaced_deployment(namespace,label_selector=label_name)
        for depl in deployments.items:
                if depl.metadata.name != datas_yaml[1]['metadata']['name']:

                # Delete it
                        print("Deleting deployment",depl.metadata.name)
                        apps_v1.delete_namespaced_deployment(namespace=namespace,name=depl.metadata.name)
                        # TODO
                        # Find the related pods and delete them immediately
        # Return the response to client

        curr_entry=pickle.loads(r.get(data['name']+":"+data['namespace']))

        curr_entry.deployment_name=datas_yaml[1]['metadata']['name']
        r.set(data['name']+":"+data['namespace'],pickle.dumps(curr_entry))
        ret="Final response"
        return Response(response=json.dumps(ret),status=200,mimetype='application.json')                

@app.route('/api/system/migrate',methods=['POST'])
def serve_migration():
        data=request.get_json()
        # TODO
        (ret,st) = migrate(old_yaml=data['oldyaml'], new_yaml=data['newyaml'], 
                AIF_name=data['name'], AIF_namespace=data['namespace'])
        print(ret)
        return Response(response=json.dumps(ret),status=st,mimetype='application.json')                

def migrate(old_yaml, new_yaml, AIF_name, AIF_namespace):
        try:
                curr_entry_pickled=r.get(AIF_name+":"+AIF_namespace)
                curr_entry=pickle.loads(curr_entry_pickled)
        except Exception:
                print("Failed to load the entry from KV store")
                return ("Failed to load the entry from KV store", 400)
        try:
                with open(new_yaml) as f:
                        datas_yaml = list(yaml.load_all(f, Loader=yaml.FullLoader))
                print(datas_yaml)
        except Exception:
                return ("Error opening the new yaml file", 400)
        try:
                v1.create_namespaced_service(namespace=AIF_namespace,body=datas_yaml[0])
                curr_entry.service_name=datas_yaml[0]['metadata']['name']
                r.set(AIF_name+":"+AIF_namespace,pickle.dumps(curr_entry))
        except Exception:
                print("Service already exists")
        # Create the deployment
        try:
                apps_v1.create_namespaced_deployment(namespace=AIF_namespace,body=datas_yaml[1])
                print("Checkpoint #1")
                # r.set(AIF_name+":"+AIF_namespace,pickle.dumps(curr_entry))
                print("The service YAML", datas_yaml[0])
                print("You can access the service via",datas_yaml[0]['spec']['ports'][0]['nodePort'])
                # Should provide the node as well
        except Exception:
                return ("Error when creating deployment", 400)

        try:
                with open(old_yaml) as f:
                        datas_yaml_old = list(yaml.load_all(f, Loader=yaml.FullLoader))
        except Exception:
                return ("Error in opening the previous file", 400)
        # try:
        print(datas_yaml_old[1])
        apps_v1.delete_namespaced_deployment(namespace=AIF_namespace,name=datas_yaml_old[1]['metadata']['name'])
        curr_entry.deployment_name=datas_yaml[1]['metadata']['name']
        # TODO 
                # Check if the creation/deletion return without error before continuing
        
        # except Exception:
                # return ("Error in deleting the previous deployment", 404)

        r.set(AIF_name+":"+AIF_namespace,pickle.dumps(curr_entry))
        return ("ok",200)

def migrate_labeled(yaml_file, AIF_namespace, AIF_name):
        # yaml_file=response_data['configFile']
        print(yaml_file)
        # svc_yaml = response_data['configFiles']['service']
        # deployment_yaml = response_data['configFiles']['service']
        curr_entry_pickled=r.get(AIF_name+":"+AIF_namespace)
        curr_entry=pickle.loads(curr_entry_pickled)
        with open(yaml_file) as f:
                datas_yaml = list(yaml.load_all(f, Loader=yaml.FullLoader))
        print(datas_yaml)
        try:
                v1.create_namespaced_service(namespace=AIF_namespace,body=datas_yaml[0])
                curr_entry.service_name=datas_yaml[0]['metadata']['name']
        except Exception:
                print("Service already exists")
        # Create the deployment
        apps_v1.create_namespaced_deployment(namespace=AIF_namespace,body=datas_yaml[1])
        print("The service YAML", datas_yaml[0])
        print("You can access the service via",datas_yaml[0]['spec']['ports'][0]['nodePort'])
        # Should provide the node as well

        #TODO wait to check where is it scheduled and advertise???


        # Remove the previous deployment
        # Get label
        label=datas_yaml[1]['metadata']['labels']['app']
        label_name="app="+label
        # Search deployments in the current namespace with the same tag?

        deployments=apps_v1.list_namespaced_deployment(AIF_namespace,label_selector=label_name)
        for depl in deployments.items:
                if depl.metadata.name != datas_yaml[1]['metadata']['name']:

                # Delete it
                        print("Deleting deployment",depl.metadata.name)
                        apps_v1.delete_namespaced_deployment(namespace=AIF_namespace,name=depl.metadata.name)
                        # TODO
                        # Find the related pods and delete them immediately
        # Return the response to client


        curr_entry.deployment_name=datas_yaml[1]['metadata']['name']
        r.set(AIF_name+":"+AIF_namespace,pickle.dumps(curr_entry))
        return
        # ret="Final response"
        # return Response(response=json.dumps(ret),status=200,mimetype='application.json')                


if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5001)


# try:
#         v1.create_namespaced_service(namespace='iccs-scheduler-apps',body=svc)
# except Exception:
#         print("Service exists propably")
# apps_v1.create_namespaced_deployment(namespace="iccs-scheduler-apps",body=data)
