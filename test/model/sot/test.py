import json
from transformers import AutoTokenizer
import numpy as np
from multiprocessing import Queue
import multiprocessing


prompt_file = "./sot_prompt_llama.json"
request = ["What are the typical types of English dishes?"]

def format_outline_prompt(outline_prompt, request):
        splits = outline_prompt.split("[ROLESWITCHING assistant:]")
        #print(splits[1])
        if len(splits) == 1:
            return splits[0].format(request=request), None
        return splits[0].format(request=request), splits[1].format(request=request)


def format_point_prompt(point_prompt, request, outline, point, point_outline):
    splits = point_prompt.split("[ROLESWITCHING assistant:]")
    if len(splits) == 1:
        return (
            splits[0].format(
                request=request,
                outline=outline,
                point=point,
                point_outline=point_outline,
            ),
            None,
        )
    return [
        split.format(
            request=request,
            outline=outline,
            point=point,
            point_outline=point_outline,
        )
        for split in splits
    ]


def test_model_inference(world_size, model_dir, model_class, batch_size, input_len, output_len, mode):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
            "weight_dir": model_dir,
            "max_total_token_num": batch_size * (input_len + output_len),
            "load_way": "HF",
            "mode": mode,
            "max_req_num": batch_size,
            "max_seq_length": (input_len + output_len),
        }

        proc = multiprocessing.Process(
            target=tppart_model_infer, args=(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue)
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    assert not ans_queue.empty()
    while not ans_queue.empty():
        assert ans_queue.get()
    return




def tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue):
    import torch
    import torch.distributed as dist
    
    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:28766", rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)
    dist.barrier()
    torch.cuda.empty_cache()
    model_part = model_class(model_kvargs)
    
    #tokenize prompt
    with open(prompt_file, "r") as rf:
        prompts = json.load(rf)
    outline_prompt = prompts["outline_prompt"]
    point_prompt = prompts["point_prompt"]
    #print("outline_prompt:",_outline_prompt)
    #print("point_prompt:",_point_prompt)
    #outline_ques提供示例，partial_answer为问题的prompt
    #outline_ques, partial_answer = format_outline_prompt(outline_prompt,request=request)
    outline_request = request.copy()
    outline_ques, partial_answer = format_outline_prompt(outline_prompt=outline_prompt,request=request[-1])
    outline_request[-1] = outline_ques
    outline_request.append(partial_answer)
    #print(len(outline_request))#list : 2
    tokenizer = AutoTokenizer.from_pretrained("/nvme/nvme2/share/share_data/llama_env/llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    #print(tokenizer.eos_token_id)
    tokenizer.padding_side = "left"
    prompt_ids = tokenizer(outline_request[0],padding="max_length",max_length=1024,return_tensors="pt").input_ids
    
    
    # warm up
    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])#batch_size , 1024
    test_data = test_data.reshape(-1)#batch_size*1024
    test_data = torch.from_numpy(test_data).cuda()

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    #input_len:sequence length 
    #b_start_loc:list[torch.int32]  every batch's start token index
    #b_seq_len:list[torch.int32]  every batch's sequence length 
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len
    
    total_token_num = input_len * batch_size
    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        test_data,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()

    for i in range(output_len):
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1
        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    if rank_id == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        print("can use req size:", model_part.req_manager.can_use_req_size)
    # warmup end
    
    
    b_req_idx = None
    b_start_loc = None
    b_seq_len = None
    
    dist.barrier()
    import time
    torch.cuda.synchronize()
    start_time = time.time()
    
    prefill_start_time = time.time()
    
    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

    total_token_num = batch_size * input_len
    #prefill
    prompt_ids = prompt_ids.reshape(-1).cuda()
    logics = model_part.forward(
        batch_size,
        total_token_num,
        input_len,
        prompt_ids,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        is_prefill=True,
    )
    
    prob_out = torch.softmax(logics, dim=-1)
    predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    predict_ids = predict_ids.detach().cpu().numpy()
    
    torch.cuda.synchronize()
    if rank_id == 0:
        print("prefill time cost:", (time.time() - prefill_start_time) * 1000)    
    output_ids = []
    for i in range(output_len):
        torch.cuda.synchronize()
        step_start = time.time()
        b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        total_token_num += batch_size
        b_seq_len += 1

        logics = model_part.forward(
            batch_size,
            total_token_num,
            input_len + i + 1,
            torch.from_numpy(predict_ids).cuda().reshape(-1),#newly input part
            b_req_idx,
            b_start_loc,
            b_seq_len,
            is_prefill=False,
        )
        prob_out = torch.softmax(logics, dim=-1)
        predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
        predict_ids = predict_ids.detach().cpu().numpy()
        
        output_ids.append(predict_ids[0][0])
        if predict_ids[0][0] == tokenizer.eos_token_id:
            break
        
        torch.cuda.synchronize()
        if i % 100 == 0 or i == output_len - 1:
            if rank_id == 0:
                print(i, "step cost time:", (time.time() - step_start) * 1000)

    torch.cuda.synchronize()
    end_time = time.time()

    if rank_id == 0:
        print("time total cost(ms):", (end_time - start_time) * 1000)
    output = tokenizer.batch_decode(output_ids,skip_special_tokens=True,spaces_between_special_tokens=False,clean_up_tokenization_spaces=True)
    print(output)
    ans_queue.put(True)

    return

def get_prompt(prompt_file):
    with open(prompt_file, "r") as rf:
        prompts = json.load(rf)
        outline_prompt = prompts["outline_prompt"]
        point_prompt = prompts["point_prompt"]
        #print("outline_prompt:",_outline_prompt)
        #print("point_prompt:",_point_prompt)
        #outline_ques提供示例，partial_answer为问题的prompt
        #outline_ques, partial_answer = format_outline_prompt(outline_prompt,request=request)
        outline_request = request.copy()
        outline_ques, partial_answer = format_outline_prompt(outline_prompt=outline_prompt,request=request[-1])
        outline_request[-1] = outline_ques
        outline_request.append(partial_answer)
        #print(len(outline_request))#list : 2
        tokenizer = AutoTokenizer.from_pretrained("/nvme/nvme2/share/share_data/llama_env/llama-2-7b-chat-hf")
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.eos_token_id)
        tokenizer.padding_side = "left"
        prompt_ids = tokenizer(outline_request[0],padding="max_length",max_length=1024,return_tensors="pt").input_ids
        #print(input_ids.shape) 1,1024
    
