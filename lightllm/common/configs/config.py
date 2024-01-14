import os
_DEFAULT_MAX_INPUT_ADD_OUTPUT_LEN = 1024 * 5

setting = {
    "max_req_total_len" : _DEFAULT_MAX_INPUT_ADD_OUTPUT_LEN,
    "nccl_port": 28765
}
setting["nccl_port"] = int(os.getenv("lightllm_nccl_port", setting['nccl_port']))