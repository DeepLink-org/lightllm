export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"

export IS_PADDING=False
# export IS_PADDING=True


 python -m lightllm.server.api_server \
    --model_dir /data/share_data/llama_model_data/llama-2-7b-chat-hf/ \
    --host 0.0.0.0 \
    --port 8888 \
    --tp 1 \
    --max_total_token_num 3200