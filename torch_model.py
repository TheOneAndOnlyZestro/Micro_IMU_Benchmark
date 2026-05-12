from torch.utils.data import DataLoader
import os
import torch
import math
from torch import nn

class ConvModel(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=9,out_channels=16, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32 * (self.seq_length - 4), out_features=7),
        )

    def forward(self, x):
        x = self.seq(x)
        return x
    
class ConvLSTMModel(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=9,out_channels=16, kernel_size=(3,1)),
            nn.ReLU(),
        )
        self.LSTM = nn.LSTM(input_size=16, hidden_size=4, batch_first=True)
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=4,out_features=7),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq_conv(x)
        x_reshaped = x.squeeze(-1).permute(0,2,1)
        x = self.LSTM(x_reshaped)[0][:,-1,:]
        x = self.final_layer(x)
        return x
    
class ConvResidualLSTMModel(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=9,out_channels=16, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32, kernel_size=(2,1)),
            nn.ReLU(),
        )
        self.seq_conv_res = nn.Sequential(
            nn.Conv2d(in_channels=9,out_channels=32, kernel_size=(4,1)),
            nn.ReLU()
        )
        self.LSTM = nn.LSTM(input_size=32, hidden_size=4, batch_first=True)
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=4,out_features=20),
            nn.ReLU(),

            nn.Linear(in_features=20,out_features=7),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.seq_conv_res(x) + self.seq_conv(x)
        x_reshaped = x.squeeze(-1).permute(0,2,1)
        x = self.LSTM(x_reshaped)[0][:,-1,:]
        x = self.final_layer(x)
        return x

class ConvLSTMModelUnrolled(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3,1)),
            nn.ReLU(),
        )
        
        # FIX: Replace nn.LSTM with nn.LSTMCell
        self.lstm_cell = nn.LSTMCell(input_size=16, hidden_size=4)
        
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=4, out_features=7),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq_conv(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1) # Shape: [Batch, Seq=8, Features=16]
        
        batch_size = x.size(0)
        seq_len = int(x.shape[1]) # Use int() to guarantee static unrolling in ONNX!
        
        # Initialize Hidden (h) and Cell (c) states to zeros
        h = torch.zeros(batch_size, 4, device=x.device)
        c = torch.zeros(batch_size, 4, device=x.device)
        
        # UNROLL THE LSTM: Manually loop through the sequence
        for i in range(seq_len):
            # Pass one time-step into the cell, update states
            h, c = self.lstm_cell(x[:, i, :], (h, c))
            
        # By the end of the loop, 'h' is the output of the very last time step!
        out = self.final_layer(h)
        return out
    
class ConvResidualLSTMModelUnrolled(nn.Module):
    def __init__(self, seq_length=10):
        super().__init__()
        self.seq_length = seq_length
        self.seq_conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1)),
            nn.ReLU(),
        )
        self.seq_conv_res = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(4,1)),
            nn.ReLU()
        )
        
        # FIX: Replace nn.LSTM with nn.LSTMCell
        self.lstm_cell = nn.LSTMCell(input_size=32, hidden_size=4)
        
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=4, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=7),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.seq_conv_res(x) + self.seq_conv(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1) # Shape: [Batch, Seq=7, Features=32]
        
        batch_size = x.size(0)
        seq_len = int(x.shape[1]) # Use int() to guarantee static unrolling
        
        # Initialize states
        h = torch.zeros(batch_size, 4, device=x.device)
        c = torch.zeros(batch_size, 4, device=x.device)
        
        # UNROLL THE LSTM
        for i in range(seq_len):
            h, c = self.lstm_cell(x[:, i, :], (h, c))
            
        # 'h' is exactly the final hidden state you need
        out = self.final_layer(h)
        return out
    
class NDenseModelModifiable(nn.Module):
    def __init__(self, n=1, seq_length=10, batch_norm=False):
        super().__init__()
        self.seq_length = seq_length
        
        layers = []
        start_units = 128
        end_units = 16
        
        in_features = 9 # Collapsed channels
        
        for i in range(n):
            if n == 1:
                units = start_units
            else:
                fraction = i / (n - 1)
                units = int(start_units * ((end_units / start_units) ** fraction))
            
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            in_features = units
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x is (B, 9, Seq, 1) -> Squeeze to (B, 9, Seq) -> Average over Seq -> (B, 9)
        x = x.squeeze(-1).mean(dim=2) 
        x = self.feature_extractor(x)
        return self.classifier(x)

class NConvModelModifiable(nn.Module):
    def __init__(self, n=1, seq_length=10, batch_norm=False):
        super().__init__()
        
        min_seq_length = 4
        max_possible_pools = int(math.log2(seq_length / min_seq_length)) if seq_length > min_seq_length else 0
        actual_pools = min(n - 1, max_possible_pools)
        
        pool_layers = []
        if actual_pools > 0:
            step = n / actual_pools
            pool_layers = [int(math.ceil((i + 1) * step)) - 1 for i in range(actual_pools)]
            
        layers = []
        in_channels = 9
        current_filters = 16
        
        for i in range(n):
            # kernel (3, 1) over (Seq, Width), padding (1, 0) keeps size same
            layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=(3, 1), padding=(1, 0)))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(current_filters))
                
            in_channels = current_filters
            
            if i in pool_layers:
                layers.append(nn.MaxPool2d(kernel_size=(2, 1)))
                current_filters = min(64, current_filters * 2)

        self.conv_blocks = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 7),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x is (B, 9, Seq, 1)
        x = self.conv_blocks(x)
        # Global Average Pooling 2D -> (B, Channels, 1, 1) -> (B, Channels)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)

class NLSTMModelModifiable(nn.Module):
    def __init__(self, n=1, seq_length=10, batch_norm=False):
        super().__init__()
        self.n = n
        self.seq_length = seq_length
        self.batch_norm = batch_norm
        
        # FIX: Replace nn.LSTM with a list of nn.LSTMCells
        self.lstm_cells = nn.ModuleList()
        if batch_norm:
            self.bns = nn.ModuleList()
            
        in_features = 9
        self.lstm_units = 16
        
        for i in range(n):
            self.lstm_cells.append(nn.LSTMCell(input_size=in_features, hidden_size=self.lstm_units))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(self.lstm_units))
            in_features = self.lstm_units
            
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_units, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (Batch, 9, Seq, 1) -> Squeeze to (Batch, 9, Seq) -> Transpose to (Batch, Seq, 9)
        x = x.squeeze(-1).transpose(1, 2)
        
        batch_size = x.size(0)
        # Use int() to guarantee static unrolling during ONNX export
        seq_len = int(x.shape[1]) 
        
        for i in range(self.n):
            # Initialize Hidden (h) and Cell (c) states to zeros for this specific layer
            h = torch.zeros(batch_size, self.lstm_units, device=x.device)
            c = torch.zeros(batch_size, self.lstm_units, device=x.device)
            
            layer_outputs = []
            
            # UNROLL THE LSTM: Manually loop through the sequence timesteps
            for t in range(seq_len):
                # Pass one time-step into the cell, update states
                h, c = self.lstm_cells[i](x[:, t, :], (h, c))
                # Store the hidden state for this timestep
                layer_outputs.append(h)
                
            # Stack the outputs back into a sequence: Shape -> (Batch, Seq, Features)
            # This becomes the input sequence 'x' for the NEXT LSTM layer
            x = torch.stack(layer_outputs, dim=1)
            
            if self.batch_norm:
                # BN1d expects (Batch, Channels, Seq), so we transpose back and forth
                x = x.transpose(1, 2)
                x = self.bns[i](x)
                x = x.transpose(1, 2)
                
        # After all layers finish, grab the very last timestep of the final sequence
        last_out = x[:, -1, :]
        return self.classifier(last_out)