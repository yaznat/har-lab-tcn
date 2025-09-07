from keras import models
from keras.layers import  Input, Conv1D, Add, Activation, BatchNormalization, Dropout, Dense, GlobalAveragePooling1D

def build_model(timesteps: int, num_features: int, num_classes: int, num_filters: int, kernel_size: int, dropout_rate: float, num_blocks: int):
    inputs = Input(shape=(timesteps, num_features))
    x = inputs
    
    for i in range(num_blocks):
        # Dilation rate controls how broadly the model can see across the data.
        # Exponential increase across blocks is a good transition from narrow to broad.
        dilation_rate = 2 ** i

        # ======= Conv1D block with residual connection and pre-activation =======
        conv = BatchNormalization()(x)
        conv = Activation("gelu")(conv)
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)(conv)
        conv = Dropout(dropout_rate)(conv)
        
        conv = BatchNormalization()(conv)
        conv = Activation("gelu")(conv)
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)(conv)
        conv = Dropout(dropout_rate)(conv)

        
        # Align channels if needed for residual
        if x.shape[-1] != num_filters: 
            x = Conv1D(num_filters, kernel_size=1, padding='same')(x)

        # Residual connection
        x = Add()([x, conv])
        x = Activation('gelu')(x)
    
    # Pool down to 2D: (batch_size, features) for classification
    x = GlobalAveragePooling1D()(x)

    # Extra Dense layer for more capacity
    x = Dense(128, activation="gelu")(x)

    # Higher dropout after Dense layer
    x = Dropout(dropout_rate * 2)(x)
    
    # Classification layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model