#!/bin/bash
find . | grep -E "(__pycache__|\.pyc|\.swp|\.pyo$)" | xargs rm -rf
