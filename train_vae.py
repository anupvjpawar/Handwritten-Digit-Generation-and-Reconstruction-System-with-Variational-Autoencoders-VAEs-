import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from vae import build_vae

(latent_dim, epochs, batch_size) = (2, 50, 64)

# Load and preprocess data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

# Build and compile VAE
vae = build_vae(latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# Train the model
vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

# Save the model
vae.save('vae_mnist.h5')
