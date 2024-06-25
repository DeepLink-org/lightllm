# export TORCH_LOGS=graph_breaks
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_REPORT_GUARD_FAILURES=1
export TORCH_SHOW_DISPATCH_TRACE=1
# export TORCHINDUCTOR_CACHE_DIR=/home/cse/zhousl/tmp

unset ASCEND_GLOBAL_LOG_LEVEL
# export ASCEND_GLOBAL_EVENT_ENABLE=0

# export DIPU_DUMP_OP_ARGS=-1
# export DIPU_MOCK_CUDA=True

# export PROFILE_COUNT=0

# export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"
# export PYTHONPATH=/data2/zhoushenglong/tmp/ext_ops:$PYTHONPATH
# export PYTHONPATH=/data2/zhoushenglong/lightllm/lightllm:$PYTHONPATH

# rm -rf ~/ascend/log/*

python test/model/test_llama2.py
