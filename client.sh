unset ASCEND_GLOBAL_LOG_LEVEL

export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True 

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"

curl http://0.0.0.0:8899/generate     \
    -X POST                           \
    -d '{"inputs":"How are you? ","parameters":{"max_new_tokens":20, "frequency_penalty":1}}' \
    -H 'Content-Type: application/json'

# curl http://0.0.0.0:8899/generate     \
#     -X POST                           \
#     -d '{"inputs":"What is the result of one plus two?","parameters":{"max_new_tokens":20, "frequency_penalty":1}}' \
#     -H 'Content-Type: application/json'

# curl http://0.0.0.0:8899/generate     \
#     -X POST                           \
#     -d '{"inputs":"What daily habits might improve mental health considering factors like sleep quality, social interaction, and exercise impact according to psychology studies done recently?   What daily habits might improve mental health considering factors like sleep quality, social interaction, and exercise impact according to psychology studies done recently? Do you knonw?","parameters":{"max_new_tokens":40, "frequency_penalty":1}}' \
#     -H 'Content-Type: application/json'
