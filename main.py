#!/usr/bin/python3

from argparse import Namespace
from tabnanny import check
from urllib import response
from regex import F
import typer
import yaml
import requests
import json

app=typer.Typer()

# Save locally for now
services=dict()
meo_address='http://192.168.1.227:31111/'
meo_address="http://localhost:5001"



@app.command()
def register(filename):
    suffix="/api/register"
    print("Create service")
    with open(filename, "r") as stream:
        try:
            vars=yaml.safe_load(stream)
            print(vars)
        except yaml.YAMLError as exc:
            print(exc)
    # Call MEO
    # if (exists(vars['name'], vars['namespace'])):
    #     print("Already exists")
    #     return

    # Add the service in the local dict
    # This will need to be in a shared KV-store, e.g., Redis

    # services[(vars['name'],vars['namespace'])] = vars

    # Call the MEO
    response = requests.post(meo_address+suffix, json=vars)

    print(response.json())
    return

@app.command()
def alter(filename):
    suffix="/api/alter"
    print("Alter service")
    with open(filename, "r") as stream:
        try:
            vars=yaml.safe_load(stream)
            print(vars)
        except yaml.YAMLError as exc:
            print(exc)

    # This is not needed
    # TOBEREMOVED (the check can be made in the alter endpoint)
    # if not (exists(vars['name'], vars['namespace'])):
    #     print("Service does not exist")
    #     return

    # Call the MEO
    response = requests.post(meo_address+suffix, json=vars)
    print(response.json())
    return


# def exists(name, namespace):
#     suffix="/api/exists"
#     response = requests.post(meo_address+suffix, json={"name":name, "namespace":namespace})
#     if response.json()=="Found":
#         return True
#     return False


@app.command()
def delete(name="",ns="",f=""):
    suffix="/api/delete"
    if f=="":
        if name != "" and ns != "":
            # Delete using name/namespace
            print("Mpla mpla")
        else:
            print("No service was provided for deletion")
            return
    else:
        if ns != "" or name != "":
            print("Since a filename is provided, name and namespace are omitted")
        # Delete using the yaml file provided
        with open(f, "r") as stream:
            try:
                vars=yaml.safe_load(stream)
                print(vars)
                name=vars['name']
                ns=vars['namespace']
            except yaml.YAMLError as exc:
                print(exc)
    
    response = requests.post(meo_address+suffix, json={"name":name, "namespace":ns})
    print(response.json())
    return


@app.command()
def invoke(name):
    if services.has_key(name):
        print("Invoke service")
    else:
        print("Service does not exist")


if __name__ == "__main__":
    app()        