#!/bin/bash

read -p "Enter project name: " projectname

echo "Initializing project folder..."
mkdir "$projectname"
mkdir "$projectname"/models
mkdir "$projectname"/datasets

cp ./datasetfunctions.py ./"$projectname"
