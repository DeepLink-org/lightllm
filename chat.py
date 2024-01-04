import argparse
import requests
import json
from datetime import datetime

class ResponseException(Exception):
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text
        message = f"响应错误, 状态码：{status_code}, 响应内容：{text}"
        super().__init__(message)


parser = argparse.ArgumentParser(description="Script as the client that initiates the request")
parser.add_argument("--url", type=str, default='http://127.0.0.1:8000/generate')
args = parser.parse_args()

headers = {'Content-Type': 'application/json'}
# When temperature is set to 0, the model uses greedy search. Thus the inference result of InternLM 7B via lightllm does not contain randomness.
# Due to code limitations, temperature can only be set to a very small number to minimize randomness.
data = {
    'inputs': 'I believe the meaning of life is',
    "parameters": {
        'temperature': 1e-5,
        'presence_penalty': 1.1,
        'frequency_penalty': 1.1,
        'max_new_tokens': 64,
    }
}
answer_reference = """to find your gift. The purpose of life is to give it away.
I believe that we are all born with a special talent, and the only way for us to be happy in this world is by sharing our talents with others. I also believe that if you don’t share your gifts, then"""

print("------start LLaMA 7B inference via lightllm------")
start = datetime.now()
response = requests.post(args.url, headers=headers, data=json.dumps(data))
end = datetime.now()
print("------end LLaMA 7B inference via lightllm------")
print(
    f'The inference time of LLaMA 7B via lightllm on the current device is: {(end - start).total_seconds():.2f} s', flush=True)

try:
    if response.status_code == 200:
        prompt = data['inputs']
        answer_output = response.json()['generated_text'][0]
        print("prompt: \n", prompt)
        print("The answer_output generated on the current device is: \n", answer_output)    
        passed = (answer_output == answer_reference)
        if passed:
            print("Successfully pass the test for the inference of LLaMA 7B via lightllm!")
        else:
            print("Fail to pass the test for the inference of LLaMA 7B!")
            print("The answer_reference generated on gpu is: \n", answer_reference)
        assert passed, "The inference result of LLaMA 7B via lightllm on the current device is not the same as the reference result generated on the gpu, please check the operator implementation of LLaMA 7B!"
    else:
        raise ResponseException(response.status_code, response.text)
except ResponseException as e:
    print(f"Response Error: {e}")
    exit(1)
