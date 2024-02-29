export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"

curl http://0.0.0.0:8888/generate \
-X POST \
-d '{"inputs":"Hello?", "parameters":{"max_new_tokens":30, "frequency_penalty":1}}' \
-H 'Content-Type: application/json'
