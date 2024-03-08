export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"
export PYTHONPATH=/data2/zhoushenglong/tmp/ext_ops:$PYTHONPATH
export PYTHONPATH=/data2/zhoushenglong/lightllm/lightllm:$PYTHONPATH

unset ASCEND_GLOBAL_LOG_LEVEL

# rm -rf ~/ascend/log/*

python test/model/test_llama2.py
