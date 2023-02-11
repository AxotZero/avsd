import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, cfg, num_layers=None):
        super().__init__()

        if num_layers is None:
            num_layers = cfg.num_gru_layers
            
        self.gru = nn.GRU(
            input_size = cfg.d_model, 
            hidden_size = cfg.d_model*2, 
            num_layers = num_layers,
            batch_first=True,
            dropout = cfg.dout_p
        )
        self.dropout = nn.Dropout(cfg.dout_p)
        self.ff = nn.Linear(cfg.d_model*2, cfg.d_model)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        
        out, _ = self.gru(x)
        return self.ff(self.dropout(out))


