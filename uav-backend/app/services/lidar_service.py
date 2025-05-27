import os
import json
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import requests

# ===============================
# CONFIGURATION
# ===============================

MODEL_PATH = "models/trained_lidar_model.pth"  # Path to the trained PyTorch model
INPUT_PLY_PATH = "data/scan_point_cloud.ply"   # Path to input LiDAR point cloud file (.ply format)
FASTAPI_URL = "http://127.0.0.1:8000/api/lidar/classified_objects"  # Backend API endpoint

# ===============================
# MODEL DEFINITIONS
# ===============================

# Shared MLP feature extractor simulating components like PointPillars and CenterPoint
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, output_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

# LSTM module for temporal fusion of extracted features
class TemporalFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension for LSTM
        out, _ = self.lstm(x)
        return self.output_layer(out[:, -1, :])  # Return output of the last timestep

# Full hybrid model combining dual feature extractors and LSTM temporal smoothing
class HybridModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.pointpillars = FeatureExtractor()
        self.centerpoint = FeatureExtractor()
        self.fusion_layer = nn.Linear(1024, 256)
        self.temporal = TemporalFusion(256, 128)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        pp_features = self.pointpillars(x)
        cp_features = self.centerpoint(x)
        combined = torch.cat((pp_features, cp_features), dim=1)
        fused = self.fusion_layer(combined)
        temporal_output = self.temporal(fused)
        return self.classifier(temporal_output)

# ===============================
# UTILITY FUNCTIONS
# ===============================

# Load and validate point cloud data
def load_point_cloud(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    return points

# Load pretrained model and set to evaluation mode
def load_model(path):
    model = HybridModel()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
    return model

# ===============================
# INFERENCE AND API POSTING
# ===============================

# Run inference and send classified data to the backend
def run_inference_and_post(model, points):
    inputs = torch.tensor(points, dtype=torch.float32)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()

    # Prepare structured result with classification info
    results = []
    for i in range(len(points)):
        results.append({
            "class_id": int(preds[i]),                     # Predicted class index
            "probabilities": probs[i].tolist(),            # Full softmax probability vector
            "coordinates": points[i].tolist()              # Original 3D point
        })

    # Save locally as a backup
    with open("classified_lidar_objects.json", "w") as f:
        json.dump({"objects": results}, f, indent=4)

    # Send to backend API
    try:
        r = requests.post(FASTAPI_URL, json={"objects": results})
        print(f"POST request sent. Status code: {r.status_code}")
    except Exception as e:
        print(f"Error sending to API: {e}")

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print("Starting LiDAR classification...")
    points = load_point_cloud(INPUT_PLY_PATH)
    model = load_model(MODEL_PATH)
    run_inference_and_post(model, points)
