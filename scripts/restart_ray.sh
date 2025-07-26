#!/bin/bash

set -ex

ray stop --force --grace-period 60 -v
env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 ray start --head --dashboard-host=0.0.0.0
