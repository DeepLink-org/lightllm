import numpy as np
from multiprocessing import Queue
import multiprocessing

import gc
import time

import logging


from torch.profiler import record_function

import acl
import torch_dipu

is_padding = False


def test_model_inference(world_size, model_dir, model_class, batch_size, input_len, output_len, mode, dynamic_compiler=False):
    ans_queue = Queue()
    workers = []
    for rank_id in range(world_size):
        model_kvargs = {
            "tp_rank": rank_id,
            "world_size": world_size,
            "weight_dir": model_dir,
            "max_total_token_num":batch_size * (input_len + output_len),
            "load_way": "HF",
            "mode": mode,
            "max_req_num": batch_size,
            "max_seq_length": (input_len + output_len),
            "dynamic_compiler": dynamic_compiler,
        }
        
        # tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue)

        proc = multiprocessing.Process(target=tppart_model_infer, args=(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue))
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

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    import os
    os.environ['LOCAL_RANK'] = str(rank_id)
    torch.cuda.set_device(rank_id)
    print(f"set device: {rank_id}", flush=True)
    
    import torch.distributed as dist
    
    dist.barrier()
    torch.cuda.empty_cache()
    gc.disable()
    print('################### 1')
    model_part = model_class(model_kvargs)
    print('################### 2')

    total_len = min(model_kvargs["max_seq_length"], input_len + output_len)

    # warm up

    test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    test_data = torch.from_numpy(test_data).cuda()

    test_data = test_data.reshape(-1)

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        b_start_loc[i] = i * input_len
        b_seq_len[i] = input_len

 
    start_pos = input_len
    prev_pos = 0

    for cur_pos in range(start_pos, total_len + 1):
        seqlen = cur_pos - prev_pos
        if seqlen > 1:
            masks = []

            total_token_num = seqlen * batch_size

            logics = model_part.forward(batch_size,
                                        total_token_num,
                                        seqlen,
                                        test_data,
                                        masks,
                                        is_padding,
                                        b_req_idx,
                                        b_start_loc,
                                        b_seq_len,
                                        is_prefill=True)

            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
            print(logics, flush=True)
            print(f"Success_prefill: {predict_ids}.", flush=True)
            # break
        else:
            masks = []

            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1
            logics = model_part.forward(batch_size, total_token_num, cur_pos, torch.from_numpy(
                predict_ids).cuda().reshape(-1), masks, is_padding, b_req_idx, b_start_loc, b_seq_len, is_prefill=False)

            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
        prev_pos = cur_pos
    print(logics, flush=True)
    print(f"Success: {predict_ids}.", flush=True)

    ans_queue.put(True)

    return

    # model_part.mem_manager.free_all()
    # model_part.req_manager.free_all()

    # if rank_id == 0:
    #     print("can use mem size:", model_part.mem_manager.can_use_mem_size)
    #     print("can use req size:", model_part.req_manager.can_use_req_size)

    # b_req_idx = None
    # b_start_loc = None
    # b_seq_len = None

    # dist.barrier()
    # torch.cuda.synchronize()
    # import time
    # start_time = time.time()

    # b_req_idx = model_part.req_manager.alloc(batch_size).int()
    # b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    # b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    # for i in range(batch_size):
    #     if is_padding:
    #         b_start_loc[i] = i * max_prompt_size
    #         b_seq_len[i] = max_prompt_size
    #     else:
    #         b_start_loc[i] = i * input_len
    #         b_seq_len[i] = input_len

    # if is_padding:
    #     start_pos = max_prompt_size
    # else:
    #     start_pos = input_len
    # prev_pos = 0

    # for cur_pos in range(start_pos, total_len + 1):
    #     seqlen = cur_pos - prev_pos
    #     if seqlen > 1:
    #         masks = []

    #         total_token_num = seqlen * batch_size

    #         prefill_start_time = time.time()
    #         # path = "/data2/zhoushenglong/lightllm/"
    #         # with torch_dipu.profiler.NativeProfile(path, with_stack=False):
    #         with torch.autograd.profiler.profile(with_stack=True, with_modules=True) as prof:
    #             logics = model_part.forward(batch_size,
    #                                         total_token_num,
    #                                         seqlen,
    #                                         test_data,
    #                                         masks,
    #                                         is_padding,
    #                                         b_req_idx,
    #                                         b_start_loc,
    #                                         b_seq_len,
    #                                         is_prefill=True)
    #         output_path = "/data2/zhoushenglong/torch_profile_prefill_t"
    #         prof.export_chrome_trace(output_path)

    #         torch.cuda.synchronize()
    #         if rank_id == 0:
    #             print("prefill time cost(ms):", (time.time() - prefill_start_time) * 1000)

    #         prob_out = torch.softmax(logics, dim=-1)
    #         predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    #         predict_ids = predict_ids.detach().cpu().numpy()

    #     else:
    #         torch.cuda.synchronize()
    #         step_start = time.time()

    #         masks = []

    #         b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
    #         total_token_num += batch_size
    #         b_seq_len += 1

    #         if True:
    #         # if False:
    #         # if cur_pos == start_pos + 1:
    #             # path = "/data2/zhoushenglong/lightllm/"
    #             # with torch_dipu.profiler.NativeProfile(path, with_stack=False):
    #             with torch.autograd.profiler.profile(with_stack=True, with_modules=True) as prof:
    #                 logics = model_part.forward(batch_size, total_token_num, cur_pos, torch.from_numpy(
    #                     predict_ids).cuda().reshape(-1), masks, is_padding, b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
    #             output_path = f"/data2/zhoushenglong/torch_profile_decode_{str(cur_pos - start_pos)}_t"
    #             prof.export_chrome_trace(output_path)
    #         else:
    #             logics = model_part.forward(batch_size, total_token_num, cur_pos, torch.from_numpy(
    #                 predict_ids).cuda().reshape(-1), masks, is_padding, b_req_idx, b_start_loc, b_seq_len, is_prefill=False)

    #         prob_out = torch.softmax(logics, dim=-1)
    #         predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    #         predict_ids = predict_ids.detach().cpu().numpy()

    #         torch.cuda.synchronize()
    #         if cur_pos % 100 == 0 or cur_pos == total_len - 1:
    #             if rank_id == 0:
    #                 print(cur_pos, "step cost time(ms):", (time.time() - step_start) * 1000)
    #     prev_pos = cur_pos
    # print(logics, flush=True)
    # print(f"Success: {predict_ids}.", flush=True)

    # torch.cuda.synchronize()
    # end_time = time.time()

    # if rank_id == 0:
    #     print("time total cost(ms):", (end_time - start_time) * 1000)

    ans_queue.put(True)

    return
