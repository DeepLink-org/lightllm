import multiprocessing
import os
import subprocess
import time
import argparse


def start_server(model_dir, tokenizer_mode, tp, max_total_token_num):
    lightllm_command = [
        "python", "-m", "lightllm.server.api_server",
        "--model_dir", model_dir,
        "--tokenizer_mode", tokenizer_mode,
        "--tp", str(tp),
        "--max_total_token_num", str(max_total_token_num),
    ]
    subprocess.run(lightllm_command)


def start_client(current_working_directory):
    subprocess.run(['python', os.path.join(
        current_working_directory, "chat.py")])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for LLaMA 7B inference via lightllm")
    parser.add_argument("--model_dir", type=str, default="/mnt/lustre/share_data/PAT/datasets/llama_7B_hf",
                        help="The model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="auto",
                        help="Tokenizer load mode, can be slow or auto")
    parser.add_argument("--tp", type=int, default=1,
                        help="Model tp parallel size, the default is 1")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="The total token nums the gpu and model can support")
    args = parser.parse_args()

    current_working_directory = os.path.dirname(os.path.abspath(__file__))
    # 启动服务器进程
    server_process = multiprocessing.Process(
        target=start_server, args=(args.model_dir, args.tokenizer_mode, args.tp, args.max_total_token_num))
    server_process.start()

    # 等待服务器启动
    time.sleep(180)

    # 启动客户端进程
    client_process = multiprocessing.Process(
        target=start_client, args=(current_working_directory,))
    client_process.start()

    # 等待客户端结束
    client_process.join()

    # 结束服务器进程
    server_process.terminate()
    server_process.join()
