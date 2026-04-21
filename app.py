import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import ops
import numpy as np
from PIL import Image
import io

# --- Configuration Parameters (from notebook) ---
characters = sorted(['D', 't', 'e', 'G', 'n', 'W', 'w', 'm', 'h', 'z', 'o', 'V', '6', 'J', 'v', '9', 's', 'H', 'Q', 'S', '5', 'i', 'C', 'f', 'u', 'B', 'g', 'd', 'q', 'j', 'l', 'Y', 'r', '3', '2', 'T', '8', '0', '1', 'P', 'X', 'c', 'b', 'Z', 'U', 'a', 'x', 'N', 'k', 'F', 'O', 'I', 'p', '7', 'E', 'L', 'K', 'R', 'y', '4', 'A', 'M'])
img_width = 200
img_height = 50
downsample_factor = 4

# --- Custom CTC Layer (required for loading the model) ---
class CTCLayer(layers.Layer):
  def __init__(self, name=None, **kwargs):
    super().__init__(name=name, **kwargs)
    self.loss_fn = keras.backend.ctc_batch_cost

  def call(self, y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = self.loss_fn(y_true, y_pred, input_length, label_length)
    self.add_loss(loss)
    return y_pred

# --- Character Mappers (recreated from configuration) ---
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# --- Prediction Decoding Function ---
def decode_batch_predictions(pred):
    input_len = ops.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        output_text.append(tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8"))
    return output_text

# --- Load the Model ---
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "model1.keras", custom_objects={'CTCLayer': CTCLayer}
    )
    # Create a prediction model from the loaded full model
    prediction_model = keras.models.Model(
        inputs=model.get_layer(name="image").input,
        outputs=model.get_layer(name="dense2").output
    )
    return prediction_model

prediction_model = load_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    img = tf.io.decode_png(tf.io.read_file(image), channels=1) # This line expects bytes or path
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = ops.image.resize(img, [img_height, img_width])
    img = ops.transpose(img, axes=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# --- Streamlit App Layout ---
st.set_page_config(layout="centered", page_title="Captcha OCR Predictor")
st.title("Captcha OCR Predictor")
st.write("Upload a captcha image and get the predicted text!")

uploaded_file = st.file_uploader("Choose a PNG image...", type="png")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Captcha', use_column_width=True)
    st.write("")

    # Read image bytes for preprocessing
    image_bytes = uploaded_file.getvalue()
    # Use tf.io.decode_png directly on the bytes
    img_tensor = tf.io.decode_png(image_bytes, channels=1)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)
    img_tensor = ops.image.resize(img_tensor, [img_height, img_width])
    img_tensor = ops.transpose(img_tensor, axes=[1, 0, 2])
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            preds = prediction_model.predict(img_tensor)
            pred_texts = decode_batch_predictions(preds)

            st.success(f"Predicted Text: {pred_texts[0]}")
else:
    st.info("Please upload a PNG image to get a prediction.")
