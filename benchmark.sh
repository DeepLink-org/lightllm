export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True

unset ASCEND_GLOBAL_LOG_LEVEL

python test/benchmark_serving.py \
    --tokenizer /data/share_data/llama_model_data/llama-2-7b-chat-hf \
    --num-prompts 20 \
    --request-rate 200
