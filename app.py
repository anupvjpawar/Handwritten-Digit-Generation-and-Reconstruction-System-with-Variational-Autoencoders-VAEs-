import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from vae import build_encoder, build_decoder

# Load VAE model
vae = load_model('vae_mnist.h5', custom_objects={'sampling': sampling})

# Build encoder and decoder
latent_dim = 2
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
decoder.set_weights(vae.get_layer('decoder').get_weights())

# Streamlit app
st.title("Handwritten Digit Generation with VAE")

st.sidebar.header("Latent Vector")
latent_vector = st.sidebar.slider("Latent Vector (x1, x2)", -3.0, 3.0, (0.0, 0.0), step=0.1)

# Generate image from latent vector
latent_vector = np.array(latent_vector).reshape(1, -1)
generated_image = decoder.predict(latent_vector)[0]

# Display generated image
st.image(generated_image, caption='Generated Digit', use_column_width=True)
