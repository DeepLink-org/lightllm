import torch
from torch.profiler import record_function

@torch.no_grad()
@record_function('apply_penalty')
def apply_penalty_v2(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids,
                        p_token_counts, p_cumsum_seq_len, p_max_len_in_batch):
    pass
    # batch = Logits.size(0)
    # for i in range(batch):
    #     cur_batch_start_index = p_cumsum_seq_len[i]
    #     cur_batch_end_index = p_cumsum_seq_len[i+1]
    #     slice = torch.arange(cur_batch_start_index, cur_batch_end_index, dtype=torch.int64,
    #                          layout = Logits.layout, device = torch.device('cpu'))
    #     cur_token_ids = p_token_ids[slice]
    #     cur_token_counts = p_token_counts[slice]
    #     cur_logits = Logits[i].index_select(0, cur_token_ids)
    #     rep_logits = torch.where(cur_logits > 0, cur_logits / repetition_penalty[i], cur_logits * repetition_penalty[i])
    #     rep_logits = rep_logits - cur_token_counts * freqency_penalty[i] - presence_penalty[i]
    #     Logits[i, cur_token_ids] = rep_logits
