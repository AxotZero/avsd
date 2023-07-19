# import torch.nn as nn

# class GRU_Linear(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.gru = nn.GRU(1, 1)
#         self.linear = nn.Linear(1, 1)
    
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         x, _ = self.gru(x)
#         x = self.linear(x)
#         return x.squeeze(-1)

# class TransformerEncoderClassifier(nn.Module):
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.transformer = nn.TransformerEncoderLayer(emb_dim, 8)
#         self.linear = nn.Linear(emb_dim, 1)
    

# class Bert(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(cfg.bert_model)
#         self.linear = nn.Linear(self.bert.config.hidden_size, 1)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         pooled_output = outputs[1]
#         logits = self.linear(pooled_output)
#         return logits

def two_sum(a, b):
    """Sum of two numbers
    Parameters
    ----------
    a : int
        first number
    b : int
        second number
        
    Returns
    -------
    int
        sum of a and b"""
    return a + b

