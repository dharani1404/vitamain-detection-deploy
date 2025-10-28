import tensorflow as tf

# Load your existing model (.h5)
print("🔹 Loading model...")
model = tf.keras.models.load_model("model/vitamin_deficiency_model.h5")

# Save a reduced-precision version (float16)
print("🔹 Saving float16 version...")
tf.keras.models.save_model(model, "model/vitamin_deficiency_model_float16.h5", save_format="h5")

# Convert to TensorFlow Lite
print("🔹 Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# You can also specify float16 quantization for extra compression
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the new TFLite model
with open("model/vitamin_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete! File saved as model/vitamin_model.tflite")
