import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_onnx_to_tflite(onnx_path, tflite_path):
    print(f"Converting: {onnx_path} -> {tflite_path}")
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert ONNX to TensorFlow
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("temp_tf_saved_model")

    # Convert TensorFlow SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_tf_saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional optimization
    tflite_model = converter.convert()

    # Save the TFLite file
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {tflite_path}\n")

def convert_all_onnx_in_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.onnx'):
            onnx_path = os.path.join(directory, file_name)
            tflite_path = os.path.join(directory, file_name.replace('.onnx', '.tflite'))
            convert_onnx_to_tflite(onnx_path, tflite_path)

if __name__ == "__main__":
    directory_to_scan = "dfn2_onnx/export"
    convert_all_onnx_in_directory(directory_to_scan)

