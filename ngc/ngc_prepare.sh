#!/bin/bash

# I use this script to install dependencies/my library or do data initialization.
# Basically, it is just a bash script that I run before executing my actual training command.

# Install dependencies
# python3 -m pip install -r requirements.txt


apt-get update || exit 1
