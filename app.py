from flask import Flask, render_template, jsonify
import os
import onnxruntime as ort
import torch
import numpy as np
from PIL import Image
from models.cnn import CustomConvNet
import time
import os
import psycopg2

app = Flask(__name__)

DATABASE_CONFIG = {
    "dbname": os.getenv("dela"),
    "user": os.getenv("postgres"),
    "password": os.getenv("postgres"),
    "host": os.getenv("0.0.0.0"),
    "port": os.getenv("5432")
}

# Path to your ONNX and PyTorch models
onnx_model_path = "model.onnx"
pt_model_path = "model_weights.pt"

# Load the PyTorch model
pt_state_dict = torch.load(pt_model_path)
pt_model = CustomConvNet(num_classes=7)
pt_model.load_state_dict(pt_state_dict)
pt_model.eval()

# Path to the test dataset
test_dataset_path = r"dataset\test"

# Initialize counters for correct and total predictions for ONNX and PyTorch
correct_predictions_onnx = 0
total_predictions_onnx = 0
correct_predictions_pt = 0
total_predictions_pt = 0

# Create a dictionary to map integers to class labels
class_label_mapping = {
    0: 'hyundai',
    1: 'lexus',
    2: 'mazda',
    3: 'mercedes',
    4: 'opel',
    5: 'toyota',
    6: 'volkswagen'
}

# Function to perform inference with ONNX Runtime and measure latency
def infer_onnx(image_tensor):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    outputs = session.run(None, {input_name: image_tensor})
    end_time = time.time()
    latency = end_time - start_time
    predicted_class_index = np.argmax(outputs[0])
    return class_label_mapping[predicted_class_index], latency

# Function to perform inference with PyTorch and measure latency
def infer_pt(image_tensor):
    image_tensor = torch.tensor(image_tensor).float()
    start_time = time.time()
    output = pt_model(image_tensor)
    end_time = time.time()
    latency = end_time - start_time
    _, predicted_class_index = torch.max(output, 1)
    return class_label_mapping[predicted_class_index.item()], latency

# Flask route to perform inference and return results
@app.route('/infer')
def infer():
    global correct_predictions_onnx, total_predictions_onnx, correct_predictions_pt, total_predictions_pt
    total_latency_onnx = 0
    total_latency_pt = 0
    for class_label in os.listdir(test_dataset_path):
        class_path = os.path.join(test_dataset_path, class_label)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                input_image = Image.open(image_path).convert('RGB')
                input_image = input_image.resize((32, 32))
                input_tensor = np.array(input_image).astype(np.float32) / 255.0
                input_tensor = np.transpose(input_tensor, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)
                
                predicted_class_onnx, latency_onnx = infer_onnx(input_tensor)
                predicted_class_pt, latency_pt = infer_pt(input_tensor)
                
                total_predictions_onnx += 1
                if predicted_class_onnx == class_label:
                    correct_predictions_onnx += 1
                total_latency_onnx += latency_onnx
                
                total_predictions_pt += 1
                if predicted_class_pt == class_label:
                    correct_predictions_pt += 1
                total_latency_pt += latency_pt
    
    accuracy_onnx = correct_predictions_onnx / total_predictions_onnx
    accuracy_pt = correct_predictions_pt / total_predictions_pt
    avg_latency_onnx = total_latency_onnx / total_predictions_onnx
    avg_latency_pt = total_latency_pt / total_predictions_pt
    
    # Return results as JSON
    results = {
        'accuracy_onnx': accuracy_onnx,
        'accuracy_pt': accuracy_pt,
        'avg_latency_onnx': avg_latency_onnx,
        'avg_latency_pt': avg_latency_pt
    }

    # Connect to PostgreSQL
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cur = conn.cursor()

    # Insert the results for the PyTorch model
    cur.execute(
        "INSERT INTO inference (model_type, nilai_akurasi, nilai_latensi) VALUES (%s, %s, %s)",
        ('pytorch', results['accuracy_pt'], results['avg_latency_pt'])
    )

    # Insert the results for the ONNX model
    cur.execute(
        "INSERT INTO inference (model_type, nilai_akurasi, nilai_latensi) VALUES (%s, %s, %s)",
        ('onnx', results['accuracy_onnx'], results['avg_latency_onnx'])
    )

    print("Data inserted")
    # Commit the transaction
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    # Return the results as JSON response
    return jsonify(results)

@app.route('/')
def home():
    print("hi")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
