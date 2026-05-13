"""
PyTorch benchmark model classes for ESP32 TFLite Micro benchmarking.

DESIGN PRINCIPLES:
- All models are regression heads (no softmax, linear final Linear)
- LSTMs use nn.LSTMCell with explicit time-step unrolling (REQUIRED for
  ONNX -> TF -> TFLite Micro export to work on ESP32)
- Static seq_length captured at __init__ for ONNX export determinism
- Input/output shape conventions match the Keras suite

INPUT CONVENTION (PyTorch):
    shape = (batch, input_features, sequence_length, 1)
    This matches your original PyTorch convention. The trailing '1' is a width dim.

OUTPUT CONVENTION:
    shape = (batch, output_units)  - regression vector
"""

import math
import torch
import torch.nn as nn


# ============================================================
# FAMILY 1: DENSE-ONLY
# ============================================================

class NDenseModel(nn.Module):
    """
    Pure Dense pipeline. Collapses sequence by mean-pooling then runs through
    a stack of Linear layers with geometrically decaying widths.
    """
    def __init__(
        self,
        n_layers: int = 2,
        seq_length: int = 10,
        start_units: int = 128,
        end_units: int = 16,
        output_units: int = 7,
        input_features: int = 9,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.input_features = input_features

        layers = []
        in_features = input_features

        for i in range(n_layers):
            if n_layers == 1:
                units = start_units
            else:
                fraction = i / (n_layers - 1)
                units = int(start_units * ((end_units / start_units) ** fraction))
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            in_features = units

        self.feature_extractor = nn.Sequential(*layers)
        # Linear regression head - NO softmax
        self.classifier = nn.Linear(in_features, output_units)

    def forward(self, x):
        # x: (B, Features, Seq, 1) -> squeeze -> (B, Features, Seq) -> mean over Seq -> (B, Features)
        x = x.squeeze(-1).mean(dim=2)
        x = self.feature_extractor(x)
        return self.classifier(x)


# ============================================================
# FAMILY 2: CONV2D-ONLY (+ optional Dense head)
# ============================================================

class NConvModel(nn.Module):
    """
    Conv2D-only architecture with optional Dense head.
    Kernel (3, 1) convolves along the sequence axis only.
    Pooling layers are distributed evenly through the conv stack.
    """
    def __init__(
        self,
        n_layers: int = 2,
        seq_length: int = 10,
        base_filters: int = 16,
        max_filters: int = 64,
        kernel_size: tuple = (3, 1),
        output_units: int = 7,
        input_features: int = 9,
        dense_head_units: list = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length

        # Distribute pooling evenly
        min_seq = 4
        max_pools = int(math.log2(seq_length / min_seq)) if seq_length > min_seq else 0
        actual_pools = min(n_layers - 1, max_pools)
        pool_indices = []
        if actual_pools > 0:
            step = n_layers / actual_pools
            pool_indices = [int(math.ceil((i + 1) * step)) - 1 for i in range(actual_pools)]

        conv_layers = []
        in_channels = input_features
        current_filters = base_filters

        for i in range(n_layers):
            # Padding (1, 0) keeps the sequence dim the same with kernel (3, 1)
            kpad = (kernel_size[0] // 2, kernel_size[1] // 2)
            conv_layers.append(
                nn.Conv2d(in_channels, current_filters, kernel_size=kernel_size, padding=kpad)
            )
            conv_layers.append(nn.ReLU())
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(current_filters))
            in_channels = current_filters
            if i in pool_indices:
                conv_layers.append(nn.MaxPool2d(kernel_size=(2, 1)))
                current_filters = min(max_filters, current_filters * 2)

        self.conv_blocks = nn.Sequential(*conv_layers)
        self.final_conv_channels = in_channels

        # Optional Dense head
        head_layers = []
        prev = in_channels
        if dense_head_units:
            for u in dense_head_units:
                head_layers.append(nn.Linear(prev, u))
                head_layers.append(nn.ReLU())
                if batch_norm:
                    head_layers.append(nn.BatchNorm1d(u))
                prev = u
        head_layers.append(nn.Linear(prev, output_units))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x: (B, Features, Seq, 1)
        x = self.conv_blocks(x)
        # Global Average Pool over spatial dims -> (B, C)
        x = x.mean(dim=(2, 3))
        return self.head(x)


# ============================================================
# FAMILY 3: LSTM-ONLY (+ Dense head)
# Uses nn.LSTMCell with explicit unrolling - REQUIRED for ESP32
# ============================================================

class NLSTMModel(nn.Module):
    """
    Stacked LSTM with explicit per-timestep unrolling via LSTMCell.

    CRITICAL: This uses nn.LSTMCell in a Python for-loop, NOT nn.LSTM.
    Only this pattern exports cleanly through ONNX -> TF -> TFLite Micro
    for ESP32 deployment.

    seq_length MUST be passed and is used as a static loop bound during export.
    """
    def __init__(
        self,
        n_layers: int = 2,
        seq_length: int = 10,
        lstm_units: int = 16,
        output_units: int = 7,
        input_features: int = 9,
        dense_head_units: list = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.batch_norm = batch_norm

        self.lstm_cells = nn.ModuleList()
        if batch_norm:
            self.bns = nn.ModuleList()

        in_features = input_features
        for i in range(n_layers):
            self.lstm_cells.append(
                nn.LSTMCell(input_size=in_features, hidden_size=lstm_units)
            )
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(lstm_units))
            in_features = lstm_units

        # Optional Dense head
        head_layers = []
        prev = lstm_units
        if dense_head_units:
            for u in dense_head_units:
                head_layers.append(nn.Linear(prev, u))
                head_layers.append(nn.ReLU())
                if batch_norm:
                    head_layers.append(nn.BatchNorm1d(u))
                prev = u
        head_layers.append(nn.Linear(prev, output_units))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x: (B, Features, Seq, 1) -> (B, Features, Seq) -> (B, Seq, Features)
        x = x.squeeze(-1).transpose(1, 2)
        batch_size = x.size(0)

        # Use int() to force a static unroll bound for ONNX
        seq_len = int(self.seq_length)

        for i in range(self.n_layers):
            h = torch.zeros(batch_size, self.lstm_units, device=x.device)
            c = torch.zeros(batch_size, self.lstm_units, device=x.device)
            outs = []
            for t in range(seq_len):
                h, c = self.lstm_cells[i](x[:, t, :], (h, c))
                outs.append(h)
            x = torch.stack(outs, dim=1)  # (B, Seq, Hidden)

            if self.batch_norm:
                # BatchNorm1d expects (B, C, L)
                x = x.transpose(1, 2)
                x = self.bns[i](x)
                x = x.transpose(1, 2)

        # Take the last timestep of the final layer
        x = x[:, -1, :]
        return self.head(x)


# ============================================================
# FAMILY 4: CONV + LSTM HYBRID (+ Dense head)
# ============================================================

class NConvLSTMModel(nn.Module):
    """
    Conv2D feature extractor + stacked unrolled LSTM + Dense head.
    The architecture family closest to your target model.

    The Conv stage convolves along the sequence axis only (kernel (k, 1))
    so the LSTM can consume the resulting sequence. Optional downsample
    halves the sequence length with stride-2 pooling.
    """
    def __init__(
        self,
        seq_length: int = 10,
        n_conv_layers: int = 2,
        n_lstm_layers: int = 2,
        conv_base_filters: int = 16,
        conv_max_filters: int = 64,
        conv_kernel_size: tuple = (3, 1),
        lstm_units: int = 16,
        output_units: int = 7,
        input_features: int = 9,
        dense_head_units: list = None,
        use_downsample: bool = False,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.batch_norm = batch_norm

        # --- Conv stage ---
        conv_layers = []
        in_channels = input_features
        current_filters = conv_base_filters
        for i in range(n_conv_layers):
            kpad = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)
            conv_layers.append(
                nn.Conv2d(in_channels, current_filters,
                          kernel_size=conv_kernel_size, padding=kpad)
            )
            conv_layers.append(nn.ReLU())
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(current_filters))
            in_channels = current_filters
            if i < n_conv_layers - 1:
                current_filters = min(conv_max_filters, current_filters * 2)

        if use_downsample:
            conv_layers.append(nn.MaxPool2d(kernel_size=(2, 1)))

        self.conv_blocks = nn.Sequential(*conv_layers)
        self.final_conv_channels = in_channels
        self.post_conv_seq = seq_length // 2 if use_downsample else seq_length
        self.lstm_units = lstm_units
        self.n_lstm_layers = n_lstm_layers
        self.seq_length = seq_length

        # --- LSTM stage (cells, unrolled) ---
        self.lstm_cells = nn.ModuleList()
        if batch_norm:
            self.lstm_bns = nn.ModuleList()
        in_feat = in_channels
        for i in range(n_lstm_layers):
            self.lstm_cells.append(
                nn.LSTMCell(input_size=in_feat, hidden_size=lstm_units)
            )
            if batch_norm:
                self.lstm_bns.append(nn.BatchNorm1d(lstm_units))
            in_feat = lstm_units

        # --- Dense head ---
        head_layers = []
        prev = lstm_units
        if dense_head_units:
            for u in dense_head_units:
                head_layers.append(nn.Linear(prev, u))
                head_layers.append(nn.ReLU())
                if batch_norm:
                    head_layers.append(nn.BatchNorm1d(u))
                prev = u
        head_layers.append(nn.Linear(prev, output_units))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        # x: (B, Features, Seq, 1)
        x = self.conv_blocks(x)
        # After conv: (B, C, Seq', 1). Squeeze width, transpose to (B, Seq', C)
        x = x.squeeze(-1).transpose(1, 2)
        batch_size = x.size(0)
        seq_len = int(self.post_conv_seq)

        for i in range(self.n_lstm_layers):
            h = torch.zeros(batch_size, self.lstm_units, device=x.device)
            c = torch.zeros(batch_size, self.lstm_units, device=x.device)
            outs = []
            for t in range(seq_len):
                h, c = self.lstm_cells[i](x[:, t, :], (h, c))
                outs.append(h)
            x = torch.stack(outs, dim=1)

            if self.batch_norm:
                x = x.transpose(1, 2)
                x = self.lstm_bns[i](x)
                x = x.transpose(1, 2)

        x = x[:, -1, :]  # last timestep
        return self.head(x)


# ============================================================
# FAMILY 5: TARGET-ARCHITECTURE REPLICA
# Parametrized version of your production model
# ============================================================

class _Unsqueeze(nn.Module):
    """Helper: adds a singleton 'channel' dim for Conv2D input."""
    def forward(self, x):
        return x.unsqueeze(1)


class _Squeeze(nn.Module):
    """Helper: removes a singleton dim from Conv2D output before BatchNorm1d."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

# ============================================================
# SWEEP CONFIG: matches keras_benchmark_models.BENCHMARK_SWEEP
# ============================================================

BENCHMARK_SWEEP = {
    "dense_only": {
        "class": NDenseModel,
        "param_grid": [
            {"n_layers": n, "start_units": 128, "end_units": 16}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
        "sequence_lengths": [5, 10, 15, 20],
        "batch_norm_options": [False, True],
    },
    "conv_only": {
        "class": NConvModel,
        "param_grid": [
            {"n_layers": n, "base_filters": 16, "max_filters": 64}
            for n in [1, 2, 3, 4, 5, 6, 8]
        ],
        "sequence_lengths": [5, 10, 15, 20],
        "batch_norm_options": [False, True],
    },
    "lstm_only": {
        "class": NLSTMModel,
        "param_grid": [
            {"n_layers": nl, "lstm_units": u, "dense_head_units": [18, 6]}
            for nl in [1, 2, 3, 4, 5]
            for u in [16, 24, 32, 48, 56, 60]
        ],
        "sequence_lengths": [5, 10, 15],
        "batch_norm_options": [False],
    },
    "conv_lstm": {
        "class": NConvLSTMModel,
        "param_grid": [
            {
                "n_conv_layers": nc,
                "n_lstm_layers": nl,
                "lstm_units": u,
                "dense_head_units": [18, 6],
                "use_downsample": ds,
            }
            for nc in [1, 2, 3]
            for nl in [1, 2, 3]
            for u in [16, 32, 56]
            for ds in [False, True]
        ],
        "sequence_lengths": [10, 15, 20],
        "batch_norm_options": [False, True],
    },
}


if __name__ == "__main__":
    # Smoke test: build each family and dry-run a forward pass
    B, F, S = 1, 9, 10

    print("=" * 60)
    print("Dense-only test (n=3, seq=10)")
    print("=" * 60)
    m1 = NDenseModel(n_layers=3, seq_length=S)
    out = m1(torch.randn(B, F, S, 1))
    print(f"  Input  : {(B, F, S, 1)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {sum(p.numel() for p in m1.parameters()):,}")

    print("\n" + "=" * 60)
    print("Conv-only test (n=3, seq=10)")
    print("=" * 60)
    m2 = NConvModel(n_layers=3, seq_length=S)
    out = m2(torch.randn(B, F, S, 1))
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {sum(p.numel() for p in m2.parameters()):,}")

    print("\n" + "=" * 60)
    print("LSTM-only test (n=2, u=56, seq=10, head=[18,6])")
    print("=" * 60)
    m3 = NLSTMModel(n_layers=2, lstm_units=56, seq_length=S, dense_head_units=[18, 6])
    out = m3(torch.randn(B, F, S, 1))
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {sum(p.numel() for p in m3.parameters()):,}")

    print("\n" + "=" * 60)
    print("Conv+LSTM test (nc=2, nl=2, u=56, seq=20, downsample=True)")
    print("=" * 60)
    m4 = NConvLSTMModel(
        seq_length=20, n_conv_layers=2, n_lstm_layers=2,
        lstm_units=56, dense_head_units=[18, 6], use_downsample=True,
    )
    out = m4(torch.randn(B, F, 20, 1))
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {sum(p.numel() for p in m4.parameters()):,}")