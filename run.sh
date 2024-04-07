#!/usr/bin/env bash
set -e

ROOT_DIR=$(cd $(dirname $0); pwd)
export TORCHINDUCTOR_CACHE_DIR=${ROOT_DIR}/compile_cache_llama
#python test.test.py 2>&1 | tee gqa.log
#python test_gqa.py 2>&1 | tee gqa.log
#python test_c10d.py 2>&1 | tee c10d.log
#python test/model/test_llama2.py 2>&1 | tee llama2.log

NODE_NUM=2
torchrun --nproc_per_node ${NODE_NUM} test_c10d.py
