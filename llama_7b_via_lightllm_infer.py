import multiprocessing
import os
import subprocess
import time
import argparse
import socket


def check_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except ConnectionRefusedError:
            return False


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
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host where the server will be running")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port on which the server will be listening")
    args = parser.parse_args()

    current_working_directory = os.path.dirname(os.path.abspath(__file__))
    # 启动服务器进程
    server_process = multiprocessing.Process(
        target=start_server, args=(args.model_dir, args.tokenizer_mode, args.tp, args.max_total_token_num))
    server_process.start()

    # 设置超时时间为10分钟
    timeout = 10 * 60
    start_time = time.time()

    # 动态等待服务器启动
    print("等待服务器启动...")
    while not check_server(args.host, args.port):
        if time.time() - start_time > timeout:
            print("等待服务器启动超时，程序退出。")
            server_process.terminate()
            server_process.join()
            exit(1)
        time.sleep(5)
    print("服务器已启动!")

    # 启动客户端进程
    client_process = multiprocessing.Process(
        target=start_client, args=(current_working_directory,))
    client_process.start()

    # 等待客户端结束
    client_process.join()

    # 结束服务器进程
    server_process.terminate()
    server_process.join()
