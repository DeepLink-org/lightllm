import torch
import numpy as np
import torch_npu


class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None

    def load_hf_weights(self, weights):
        """
        load weights
        """
        pass

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        raise Exception("must verify weights load ok")
        pass

    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).npu()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).npu(self.tp_rank_)
