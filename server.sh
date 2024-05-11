# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export TORCHDYNAMO_REPORT_GUARD_FAILURES=1

unset ASCEND_GLOBAL_LOG_LEVEL
# export ASCEND_GLOBAL_EVENT_ENABLE=0

export PROFILE_COUNT=0

export DIPU_DUMP_OP_ARGS=-1
export DIPU_MOCK_CUDA=True

export DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS="rsqrt.out,mm,linear,_softmax.out"

export IS_PADDING=False
# export IS_PADDING=True

python -m lightllm.server.api_server \
   --model_dir /data/share_data/llama_model_data/llama-2-7b-chat-hf/ \
   --host 0.0.0.0 \
   --port 8899 \
   --tp 1 \
   --max_req_input_len 256 \
   --max_req_total_len 512 \
   --max_total_token_num 15000 \
   --tokenizer_mode auto
