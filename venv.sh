#!/usr/bin/env bash
# This is used to install the python virtual env

rm -rf venv &&
python -m venv venv &&
venv/bin/pip install -r requirements.txt
