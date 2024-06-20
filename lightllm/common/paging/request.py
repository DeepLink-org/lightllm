
from dataclasses import dataclass, field

from lightllm.common.paging.block import LogicalTokenBlocks

@dataclass
class Request:
    def __init__(self, seq_len, block_size):
        self._seq_len = seq_len
        self.block_size = block_size
        self.logical_blocks = LogicalTokenBlocks()

    def update_seq_len(self, seq_len):
        self._seq_len = seq_len

    def seq_len(self):
        return self._seq_len
    
    def num_blocks(self):
        """num blocks."""
        return len(self.logical_blocks)
