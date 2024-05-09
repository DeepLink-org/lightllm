# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, Union

import numpy as np

from lightllm.common.paging.block import _div_up
from lightllm.common.paging.block_manager.base_block_manager import BaseBlockManager
from lightllm.common.paging.request import Request


BlockTable = np.ndarray


class DefaultBlockManager(BaseBlockManager):
    """Manage the usage of blocks, generate block tables.

    Args:
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    @classmethod
    def num_required_blocks(cls,
                            req: Request):
        """get num required blocks."""
        num_tokens = req.seq_len()
        num_all_blocks = _div_up(num_tokens, req.block_size)
        return max(0, num_all_blocks - len(req.logical_blocks))

    @classmethod
    def last_block_size(cls, req: Request) -> int:
        """get last block size."""
        num_blocks = len(req.logical_blocks)
        if num_blocks == 0:
            return 0
        return req.seq_len % req.block_size

    def can_allocate(self, req: Request):
        """Return if physical block can be allocated for given request."""
        num_required_blocks = self.num_required_blocks(req)
        num_free_phy = self.get_num_free_gpu_blocks()
        return num_required_blocks <= num_free_phy

    def allocate_msg(self, req: Request):
        """Allocate physical blocks for given message according to logical
        blocks."""
        logical_blocks = req.logical_blocks
        num_required_blocks = self.num_required_blocks(req)
        if num_required_blocks > 0:
            blocks = self.allocator.allocate(num_required_blocks, 'gpu')
            logical_blocks.append(blocks)

   
    def free(self, req: Request):
        """Free all physical blocks allocated for the session."""
        self.allocator.free(req.logical_blocks.get_real_blocks())
        req.logical_blocks.reset()

    