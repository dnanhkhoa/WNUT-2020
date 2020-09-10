#!/bin/bash

find . -iname "__pycache__" -type d -exec rm -rf "{}" \;
find . -iname "*.pyc" -type f -exec rm -rf "{}" \;
find . -iname ".DS_Store" -type f -exec rm -rf "{}" \;
