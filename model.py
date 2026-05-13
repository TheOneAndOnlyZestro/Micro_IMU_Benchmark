"""
Keras benchmark model generators for ESP32 TFLite Micro benchmarking.

DESIGN PRINCIPLES:
- All models are regression heads (no softmax, linear final Dense)
- All LSTMs use unroll=True (REQUIRED for ESP32)
- Static batch_size=1 supported via batch_input_shape for TFLite Micro
- Generator pattern: every architecture function returns a builder function

INPUT CONVENTION:
    shape = (batch_size, sequence_length, 1, input_features)
    The middle '1' is a width dim used by Conv2D with kernel (k, 1).

OUTPUT CONVENTION:
    shape = (batch_size, output_units)  - regression vector
"""

import keras
import math


# ============================================================
# FAMILY 1: DENSE-ONLY
# Varying depth and width to isolate Dense-layer compute cost.
# ============================================================

def create_dense_model_generator(
    n_layers: int = 2,
    start_units: int = 128,
    end_units: int = 16,
    output_units: int = 7,
    input_features: int = 9,
    batch_size=None,
):
    """
    Pure Dense pipeline after a GlobalAveragePooling collapse over the sequence.
    Layer widths decay geometrically from start_units to end_units.

    Use this to isolate the cost of fully-connected math without temporal
    or convolutional ops in the way.
    """
    def build(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential(name=f"dense_n{n_layers}_s{sequence_length}")
        model.add(keras.layers.InputLayer(
            batch_input_shape=(batch_size, sequence_length, 1, input_features)
        ))
        # Collapse to (B, Seq, Features) then pool over time -> (B, Features)
        model.add(keras.layers.Reshape((sequence_length, input_features)))
        model.add(keras.layers.GlobalAveragePooling1D())

        for i in range(n_layers):
            if n_layers == 1:
                units = start_units
            else:
                fraction = i / (n_layers - 1)
                units = int(start_units * ((end_units / start_units) ** fraction))
            model.add(keras.layers.Dense(units, activation='relu'))
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())

        # Linear regression head - NO softmax for regression
        model.add(keras.layers.Dense(units=output_units, activation=None))
        return model

    return build


# ============================================================
# FAMILY 2: CONV2D-ONLY (+ optional Dense head)
# Varying Conv depth/width to isolate convolution cost.
# ============================================================

def create_conv_model_generator(
    n_layers: int = 2,
    base_filters: int = 16,
    max_filters: int = 64,
    kernel_size: tuple = (3, 1),
    output_units: int = 7,
    input_features: int = 9,
    dense_head_units: list = None,  # e.g. [32, 16] for two Dense layers before output
    batch_size=None,
):
    """
    Conv2D-only architecture with optional Dense head.
    Performs pooling along the sequence axis at evenly spaced intervals,
    doubling filter count after each pool (capped at max_filters).

    The kernel shape (k, 1) only convolves along the sequence axis -
    matches your target architecture's temporal Conv1d pattern.
    """
    def build(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential(name=f"conv_n{n_layers}_s{sequence_length}")
        model.add(keras.layers.InputLayer(
            batch_input_shape=(batch_size, sequence_length, 1, input_features)
        ))

        # Distribute pooling layers evenly across the conv stack
        min_seq_length = 4
        if sequence_length > min_seq_length:
            max_possible_pools = int(math.log2(sequence_length / min_seq_length))
        else:
            max_possible_pools = 0
        actual_pools = min(n_layers - 1, max_possible_pools)

        pool_layers = []
        if actual_pools > 0:
            step = n_layers / actual_pools
            pool_layers = [int(math.ceil((i + 1) * step)) - 1 for i in range(actual_pools)]

        current_filters = base_filters
        for i in range(n_layers):
            model.add(keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=kernel_size,
                padding="same",
                activation='relu',
            ))
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())
            if i in pool_layers:
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
                current_filters = min(max_filters, current_filters * 2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # Optional Dense head before regression output
        if dense_head_units:
            for u in dense_head_units:
                model.add(keras.layers.Dense(u, activation='relu'))
                if batch_normalization:
                    model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=output_units, activation=None))
        return model

    return build


# ============================================================
# FAMILY 3: LSTM-ONLY (+ optional Dense head)
# Varying LSTM depth, hidden units, and Dense head width.
# ============================================================

def create_lstm_model_generator(
    n_layers: int = 2,
    lstm_units: int = 16,
    output_units: int = 7,
    input_features: int = 9,
    dense_head_units: list = None,  # e.g. [18, 6] mirrors your target's regressor
    batch_size=None,
):
    """
    Stacked-LSTM architecture with optional Dense regression head.

    CRITICAL: unroll=True on every LSTM. This is REQUIRED for ESP32 inference
    to work at all in TFLite Micro. The unrolled graph is what gets converted.
    """
    def build(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential(name=f"lstm_n{n_layers}_u{lstm_units}_s{sequence_length}")
        model.add(keras.layers.InputLayer(
            batch_input_shape=(batch_size, sequence_length, 1, input_features)
        ))
        model.add(keras.layers.Reshape((sequence_length, input_features)))

        for i in range(n_layers):
            return_seq = (i < n_layers - 1)
            model.add(keras.layers.LSTM(
                units=lstm_units,
                return_sequences=return_seq,
                unroll=True,  # MANDATORY for ESP32 TFLite Micro
            ))
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())

        # Optional Dense head (your target uses [18, 6, 2])
        if dense_head_units:
            for u in dense_head_units:
                model.add(keras.layers.Dense(u, activation='relu'))
                if batch_normalization:
                    model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=output_units, activation=None))
        return model

    return build


# ============================================================
# FAMILY 4: CONV + LSTM HYBRID (+ Dense head)
# Mirrors the structure of your target architecture.
# ============================================================

def create_conv_lstm_model_generator(
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
    batch_size=None,
):
    """
    Conv2D feature extractor + stacked LSTM + Dense head.

    This is the architecture family closest to your target model. Vary
    n_conv_layers, n_lstm_layers, and lstm_units to sweep around the target's
    sweet spot (2 conv stages, 2 LSTM layers, 56 hidden units).

    The Conv block reduces (B, Seq, 1, F) along the feature axis to (B, Seq', 1, Cout),
    which then gets reshaped to (B, Seq', Cout) for LSTM input.

    use_downsample: adds a strided pooling after conv block, mirroring your
    target's Downsample_Block.
    """
    def build(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential(
            name=f"conv{n_conv_layers}_lstm{n_lstm_layers}u{lstm_units}_s{sequence_length}"
        )
        model.add(keras.layers.InputLayer(
            batch_input_shape=(batch_size, sequence_length, 1, input_features)
        ))

        # === CONV STAGE ===
        # We do NOT pool sequence dim inside the conv stack here, because the
        # LSTM consumes the sequence directly. Filter count grows linearly.
        current_filters = conv_base_filters
        for i in range(n_conv_layers):
            model.add(keras.layers.Conv2D(
                filters=current_filters,
                kernel_size=conv_kernel_size,
                padding="same",
                activation='relu',
            ))
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())
            # Increase filters mid-stack (capped)
            if i < n_conv_layers - 1:
                current_filters = min(conv_max_filters, current_filters * 2)

        # === RESHAPE FOR LSTM ===
        # Drop the width-1 dim. After this, sequence_length may be halved if
        # downsample was used; tf will infer the new shape.
        # Keras handles this with target_shape=(-1, filters) doing flatten-on-time.
        # Better: explicit Reshape using the known final spatial size.
        post_conv_seq = sequence_length // 2 if use_downsample else sequence_length
        model.add(keras.layers.Reshape((post_conv_seq, current_filters)))

        # === LSTM STAGE ===
        for i in range(n_lstm_layers):
            return_seq = (i < n_lstm_layers - 1)
            model.add(keras.layers.LSTM(
                units=lstm_units,
                return_sequences=return_seq,
                unroll=True,  # MANDATORY for ESP32
            ))
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())

        # === DENSE HEAD ===
        if dense_head_units:
            for u in dense_head_units:
                model.add(keras.layers.Dense(u, activation='relu'))
                if batch_normalization:
                    model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=output_units, activation=None))
        return model

    return build


# ============================================================
# SWEEP CONFIG: standardized parameter grid for the benchmark
# ============================================================



if __name__ == "__main__":
    # Smoke test: build one model from each family and print its summary
    print("=" * 60)
    print("DENSE-only example (n=3, seq=10)")
    print("=" * 60)
    m1 = create_dense_model_generator(n_layers=3)(sequence_length=10)
    m1.summary()

    print("\n" + "=" * 60)
    print("CONV-only example (n=3, seq=15)")
    print("=" * 60)
    m2 = create_conv_model_generator(n_layers=3)(sequence_length=15)
    m2.summary()

    print("\n" + "=" * 60)
    print("LSTM-only example (n=2, u=56, seq=10)")
    print("=" * 60)
    m3 = create_lstm_model_generator(n_layers=2, lstm_units=56,
                                     dense_head_units=[18, 6])(sequence_length=10)
    m3.summary()

    print("\n" + "=" * 60)
    print("CONV+LSTM example (nc=2, nl=2, u=56, seq=20)")
    print("=" * 60)
    m4 = create_conv_lstm_model_generator(
        n_conv_layers=2, n_lstm_layers=2, lstm_units=56,
        dense_head_units=[18, 6]
    )(sequence_length=20, batch_normalization=True)
    m4.summary()