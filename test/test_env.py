import torch
import torch_dipu

torch.cuda.set_device(3)

# tril
class OpModule(torch.nn.Module):
    def forward(self, ones):
        #res_default = torch.ops.aten.tril.default(ones)
        #res_default = ones[4:8]
        res_default = torch.ops.aten.split.Tensor(ones, 4)
        return res_default

#ones = torch.ones(16, 16, dtype=torch.float32, device='dipu')
ones = torch.randn(8, 32, 32, dtype=torch.float32, device='cuda:3')
model = OpModule()

print("test 1", flush=True)
compiled_model = torch.compile(model, backend='ascendgraph')
print("test 2", flush=True)

out = compiled_model(ones)
print("test 3", flush=True)

print('out: ', out)
print('out.shape:', out[1].shape)
