import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Encoder
def build_encoder(latent_dim):
    encoder_input = layers.Input(shape=(28, 28, 1))
    x = layers.Flatten()(encoder_input)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    encoder = models.Model(encoder_input, [z_mean, z_log_var], name="encoder")
    return encoder

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Decoder
def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(28 * 28, activation='sigmoid')(x)
    decoder_output = layers.Reshape((28, 28, 1))(x)
    decoder = models.Model(latent_inputs, decoder_output, name="decoder")
    return decoder

# VAE model
def build_vae(latent_dim):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    
    encoder_input = layers.Input(shape=(28, 28, 1))
    z_mean, z_log_var = encoder(encoder_input)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    decoder_output = decoder(z)
    
    vae = models.Model(encoder_input, decoder_output, name="vae")
    
    # Loss function
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(encoder_input, decoder_output))
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    
    return vae
