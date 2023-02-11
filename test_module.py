# import torch

# def get_sent_indices(text_indices, sent_start_idx=0, sent_end_idx=1):
#     sent_indices = (text_indices == sent_start_idx).long() 
#     for i in range(1, text_indices.size()[1]):
#         sent_indices[:, i] += sent_indices[:, i-1]
#     sent_indices -= 1
#     return sent_indices


# def get_mask(text_indices, sent_indices):
#     bs, seq_len = text_indices.size()
#     mask = torch.ones((seq_len, seq_len), dtype=torch.bool).triu(1)
#     for i in 

    
    


