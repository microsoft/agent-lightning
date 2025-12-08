#!/bin/bash

set -ex

ray stop -v --force --grace-period 60
ps aux

export http_proxy="http://172.31.255.10:8888"
export https_proxy="http://172.31.255.10:8888"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export no_proxy="$NO_PROXY"

env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 RAY_USAGE_STATS_ENABLED=0 \
    http_proxy="$http_proxy" \
    https_proxy="$https_proxy" \
    HTTP_PROXY="$HTTP_PROXY" \
    HTTPS_PROXY="$HTTPS_PROXY" \
    NO_PROXY="$NO_PROXY" \
    no_proxy="$no_proxy" \
    ray start --head --include-dashboard=false --disable-usage-stats
