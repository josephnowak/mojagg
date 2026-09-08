#!/usr/bin/env bash
set -u
cd /mnt/c/Users/usuario/PycharmProjects/PythonProject/mojo-group-by
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
export PYTHONPATH=python
python benchmarks/gufunc_driver/run.py
