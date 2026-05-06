import keras
import math

def create_dense_n_model(n: int = 1, batch_size=None):
    def create_dense_model(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, sequence_length, 1, 9)))
        
        # Collapse spatial dimensions to feed purely feature data into Dense layers
        model.add(keras.layers.Reshape((sequence_length, 9)))
        model.add(keras.layers.GlobalAveragePooling1D())
        
        start_units = 128
        end_units = 16
        
        for i in range(n):
            if n == 1:
                units = start_units
            else:
                # Smoothly decay units from 128 down to 16 across 'n' layers
                fraction = i / (n - 1)
                units = int(start_units * ((end_units / start_units) ** fraction))
            
            model.add(keras.layers.Dense(units, activation='relu'))
            
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())
        
        model.add(keras.layers.Dense(units=7, activation='softmax'))
        return model

    return create_dense_model    


def create_conv_n_model(n: int = 1, batch_size=None):
    def create_conv_model(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, sequence_length, 1, 9)))
        
        # 1. Calculate how many times we can safely halve the sequence length
        min_seq_length = 4  # Stop pooling when sequence length drops to this size
        if sequence_length > min_seq_length:
            max_possible_pools = int(math.log2(sequence_length / min_seq_length))
        else:
            max_possible_pools = 0
            
        # 2. We cannot pool more times than we have layers (minus 1)
        actual_pools = min(n - 1, max_possible_pools)
        
        # 3. Distribute the pooling layers evenly across the 'n' layers
        pool_layers = []
        if actual_pools > 0:
            step = n / actual_pools
            # Example: if n=10 and actual_pools=3, it pools after layers 3, 6, and 9
            pool_layers = [int(math.ceil((i + 1) * step)) - 1 for i in range(actual_pools)]
            
        current_filters = 16  # Base channels
        
        for i in range(n):
            model.add(keras.layers.Conv2D(
                filters=current_filters, 
                kernel_size=(3, 1), 
                padding="same", 
                activation='relu'
            ))
            
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())
                
            # If this specific layer index was chosen for downsizing
            if i in pool_layers:
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 1)))
                
                # Increase channels for the next layer (cap at 64 or 128 to save ESP32 RAM)
                current_filters = min(64, current_filters * 2)
        
        # 4. Use GlobalAveragePooling instead of Flatten to ensure 
        # consistency in the Dense layer regardless of leftover sequence length
        model.add(keras.layers.GlobalAveragePooling2D())
        
        model.add(keras.layers.Dense(units=7, activation='softmax'))

        return model
    return create_conv_model

def create_lstm_n_model(n: int = 1, batch_size=None):
    def create_lstm_model(sequence_length: int, batch_normalization: bool = False):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(batch_input_shape=(batch_size, sequence_length, 1, 9)))
        model.add(keras.layers.Reshape((sequence_length, 9)))
        
        # Keep units fixed to isolate the benchmark to just the depth ('n')
        lstm_units = 16 
        
        for i in range(n):
            # All layers EXCEPT the last one must return the sequence
            return_seq = (i < n - 1)
            
            model.add(keras.layers.LSTM(
                units=lstm_units, 
                return_sequences=return_seq, 
                unroll=True # Crucial for ESP32 inference speed
            ))
            
            if batch_normalization:
                model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=7, activation='softmax')) 
        return model
    
    return create_lstm_model

def create_residual_conv_lstm_model(sequence_length: int = 10, batch_size=None):
    # Use batch_shape here to allow enforcing a static batch size
    inputs = keras.Input(batch_shape=(batch_size, sequence_length, 1, 9))
    
    conv_layer_1 = keras.layers.Conv2D(filters=16, kernel_size=(3,1), activation='relu')
    conv_layer_2 = keras.layers.Conv2D(filters=32, kernel_size=(2,1), activation='relu')
    conv_layer_res = keras.layers.Conv2D(filters=32, kernel_size=(4,1), activation='relu')
    
    # Residual
    x = conv_layer_2(conv_layer_1(inputs)) + conv_layer_res(inputs)
    x = keras.layers.Reshape(((sequence_length - 3 + 1) - 2 + 1, 32))(x)
    
    lstm_layer_1 = keras.layers.LSTM(units=4, unroll=True)
    x = lstm_layer_1(x)
    
    x = keras.layers.Dense(units=20, activation='relu')(x)
    outputs = keras.layers.Dense(units=7)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model




