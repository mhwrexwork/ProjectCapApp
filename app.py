import streamlit as st
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
from PIL import Image
import io


# --- Configuration Parameters (from notebook) ---
characters = sorted(['F', '6', '1', 'B', 'a', 'L', 'm', '2', '7', 'k', 'c', 'v', 'p', 'z', '4', 'N', 'O', 'I', 'y', 'g', 'P', '3', 'K', 'V', 'E', 'U', 'T', 'j', 'Q', 'J', 'l', '0', 'r', 'i', 's', 'n', 'M', 'W', 'q', 'S', 'G', 'f', 'x', 'e', 't', 'd', 'Y', 'A', 'R', 'w', 'b', '5', 'h', 'o', 'H', '8', 'u', 'Z', 'C', 'X', 'D', '9'])
img_width = 200
img_height = 50
downsample_factor = 4

# --- Custom CTC Layer (required for loading the model) ---
@keras.saving.register_keras_serializable()
class CTCLayer(layers.Layer):
  def __init__(self, name=None, **kwargs):
    super().__init__(name=name, **kwargs)
    self.loss_fn = keras.ops.ctc_loss

  def call(self, y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.fill([batch_len], tf.cast(tf.shape(y_pred)[1], dtype="int64"))
    label_length = tf.fill([batch_len], tf.cast(tf.shape(y_true)[1], dtype="int64"))

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

# --- Model Architecture (rebuilt for prediction) ---
def build_prediction_model_architecture():
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name= "image", dtype="float32"
    )

    x = layers.Conv2D(
        32,
        (3, 3),
        activation = "relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1"
    )(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation = "relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((img_width // downsample_factor), (img_height // downsample_factor) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    num_classes = char_to_num.vocabulary_size() + 1
    output_dense2 = layers.Dense(num_classes, activation="softmax", name="dense2")(x)

    # The prediction model only needs the image input and the final dense layer output
    prediction_model = keras.models.Model(inputs=input_img, outputs=output_dense2)
    return prediction_model

# --- Load the Model (and its weights) ---
@st.cache_resource
def load_model():
    # 1. Build the prediction model architecture
    prediction_model = build_prediction_model_architecture()

    # 2. Load the weights from the saved model file
    # We first load the full model (which includes CTCLayer) to get the weights correctly
    # and then apply those weights to our prediction_model_architecture
    full_saved_model = keras.models.load_model(
        "clean_model.keras", custom_objects={'CTCLayer': CTCLayer}, compile=False
    )
    prediction_model.set_weights(full_saved_model.get_weights())

    return prediction_model

prediction_model = load_model()

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
