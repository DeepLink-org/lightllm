import os
import torch_dipu
import torch.distributed as dist
import torch
import torch.distributed._functional_collectives as funcol
import lightllm.patch_lightllm_ops

def dist_func(in_tensor, weight, pg):
    logic_batch = torch.mm(weight, in_tensor)
    #64x128 128x256 64x256 --> 128*256
    gather_result = funcol.all_gather_tensor(logic_batch, 0, pg)
    # print(f"rank {local_rank} gather_result: {gather_result}")
    #all_gather_list = [torch.zeros(2, dtype=torch.float32, device='cuda'), torch.zeros(2, dtype=torch.float32, device='cuda')]
    #dist.all_gather(all_gather_list, input_tensor)
    #gather_result = torch.stack(all_gather_list, dim=0)
    return gather_result


def test_dist_gather():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    local_tensor = torch.tensor([world_size*local_rank, world_size*local_rank+1], dtype=torch.float32, device='cuda')
    weight = torch.randn([64, 128], dtype=torch.float16, device='cuda')
    in_tensor = torch.randn([128, 256], dtype=torch.float16, device='cuda')

    default_pg = None
    default_pg = dist.distributed_c10d._get_default_group()
    #import pdb;pdb.set_trace()

    print(f"rank {local_rank} world_size {world_size} local_tensor: {local_tensor.shape}")
    compiled_dist_func = torch.compile(dist_func, backend="ascendgraph")

    result_compile = compiled_dist_func(in_tensor, weight, default_pg)
    print(f"rank {local_rank} dist_result_compile: {result_compile.shape}  {result_compile}")


    #dist.destroy_process_group()
    #dist.init_process_group(backend="mpi")
    #default_pg = dist.distributed_c10d._get_default_group()
    weight = weight.cpu().float()
    in_tensor = in_tensor.cpu().float()
    ref = torch.mm(weight, in_tensor)
    return (torch.allclose(result_compile[local_rank*64:(local_rank+1)*64,:].cpu().float(),ref, atol= 1e-2, rtol=1e-2))

    #result_ref = funcol.all_gather_tensor(ref, 0, default_pg)
    #print(f"rank {local_rank} dist_result: {result_ref.shape}  {result_ref}")


if __name__ == "__main__":
    if test_dist_gather():
        print("gather test passed")
    else:
        print("gather test failed")
