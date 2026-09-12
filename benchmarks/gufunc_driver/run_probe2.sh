#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
export PYTHONPATH=python
python benchmarks/gufunc_driver/probe_pools.py
