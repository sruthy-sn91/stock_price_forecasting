import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (seq_len, batch_size, d_model)
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # For regression output

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_features)
        """
        # Permute to (seq_len, batch_size, num_features)
        x = x.permute(1, 0, 2)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Take the last time step's output, shape: (batch_size, d_model)
        last_step = encoded[-1, :, :]
        
        # Final linear layer
        out = self.fc_out(last_step)
        return out
