import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
import keras

@keras.saving.register_keras_serializable(package="models")
class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder_dense1 = layers.Dense(128, activation='relu', name='encoder_dense1', kernel_regularizer=L2(1e-4))
        self.encoder_dropout1 = layers.Dropout(0.2, name='encoder_dropout1')
        self.encoder_dense2 = layers.Dense(latent_dim, activation='relu', name='encoder_latent', kernel_regularizer=L2(1e-4))

        # Decoder layers
        self.decoder_dense1 = layers.Dense(128, activation='relu', name='decoder_dense1', kernel_regularizer=L2(1e-4))
        self.decoder_dropout1 = layers.Dropout(0.2, name='decoder_dropout1')
        self.decoder_dense2 = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')

    def call(self, x):
        # Encoder
        enc1 = self.encoder_dense1(x)
        enc1_drop = self.encoder_dropout1(enc1)
        latent = self.encoder_dense2(enc1_drop)
        
        # Decoder
        dec1 = self.decoder_dense1(latent)
        dec1_drop = self.decoder_dropout1(dec1)
        reconstructed = self.decoder_dense2(dec1_drop)
        return reconstructed
    
    def encode(self, x):
        """Encode input to latent representation"""
        enc1 = self.encoder_dense1(x)
        enc1_drop = self.encoder_dropout1(enc1)
        latent = self.encoder_dense2(enc1_drop)
        return latent
    
    def decode(self, latent):
        """Decode latent representation to output"""
        dec1 = self.decoder_dense1(latent)
        dec1_drop = self.decoder_dropout1(dec1)
        reconstructed = self.decoder_dense2(dec1_drop)
        return reconstructed
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        })
        return config

def build_autoencoder_model(input_dim, latent_dim, learning_rate=1e-4, weight_decay=1e-3):
    model = Autoencoder(input_dim, latent_dim)
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
