import numpy as np
import heapq
from collections import Counter
from rdp import rdp
import zlib  # For binary compression

### GPS Compression Functions ###
def delta_encode(data):
    return np.diff(data, prepend=data[0])

def delta_decode(delta_data):
    return np.cumsum(delta_data)

def build_huffman_tree(freqs):
    heap = [[weight, [symbol, ""]] for symbol, weight in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))

def huffman_encode(data):
    freqs = Counter(data)
    huffman_tree = build_huffman_tree(freqs)
    encoded_data = ''.join(huffman_tree[value] for value in data)
    return encoded_data, huffman_tree

def downsample_gps(points, grid_size=0.0002):
    unique_points = np.unique(points // grid_size * grid_size, axis=0)
    return unique_points

def compress_gps(latitude, longitude):
    gps_points = np.column_stack((latitude, longitude))
    downsampled_gps = rdp(gps_points, epsilon=1e-4)

    simplified_lat, simplified_lon = downsampled_gps[:, 0], downsampled_gps[:, 1]
    delta_lat = delta_encode(simplified_lat)
    delta_lon = delta_encode(simplified_lon)

    encoded_lat, lat_tree = huffman_encode(delta_lat)
    encoded_lon, lon_tree = huffman_encode(delta_lon)

    compressed_lat = zlib.compress(encoded_lat.encode())
    compressed_lon = zlib.compress(encoded_lon.encode())

    return compressed_lat, compressed_lon

### IMU Compression ###
def run_length_encode(data):
    values, counts = [], []
    prev = data[0]
    count = 1
    for i in range(1, len(data)):
        if data[i] == prev:
            count += 1
        else:
            values.append(prev)
            counts.append(count)
            prev = data[i]
            count = 1
    values.append(prev)
    counts.append(count)
    return values, counts

def quantize(data, levels=16):
    min_val, max_val = np.min(data), np.max(data)
    step = (max_val - min_val) / levels
    quantized = np.round((data - min_val) / step).astype(int)
    return quantized, min_val, step

def compress_imu(imu_data):
    """Ensure imu_data is a NumPy array before processing."""
    if not isinstance(imu_data, np.ndarray):
        imu_data = np.array(imu_data)

    quantized_imu, min_val, step = quantize(imu_data.flatten(), levels=128)
    rle_values, rle_counts = run_length_encode(quantized_imu)
    return rle_values, rle_counts, min_val, step

### LiDAR Compression ###
def compress_lidar(lidar_points, voxel_size=0.1):
    """
    Compresses LiDAR data using voxel grid downsampling and Huffman encoding.
    """
    # Voxel grid downsampling
    grid = (lidar_points / voxel_size).astype(int)
    unique_voxels = np.unique(grid, axis=0)
    downsampled_points = unique_voxels * voxel_size

    # Flatten and Huffman encode
    flat_points = downsampled_points.flatten().astype(int)
    encoded_data, huffman_tree = huffman_encode(flat_points)

    # Compress using zlib
    compressed_data = zlib.compress(encoded_data.encode())
    return compressed_data, huffman_tree

### GA Waypoint Compression ###
def compress_ga_waypoints(waypoints):
    """
    Compresses GA waypoints using delta encoding, quantization, and LZMA compression.
    """
    # Delta encoding
    delta_encoded = delta_encode(waypoints)

    # Quantization
    quantized, min_val, step = quantize(delta_encoded, levels=256)

    # LZMA compression
    serialized_data = quantized.tobytes()
    compressed_data = zlib.compress(serialized_data)

    return compressed_data, min_val, step

### Data Prioritization Based on Bandwidth ###
def prioritize_data(data, bandwidth_limit):
    """
    Prioritizes telemetry data based on available bandwidth.
    """
    # Assign priorities (e.g., IMU > GPS > LiDAR)
    priority_order = ["imu", "gps", "lidar"]
    prioritized_data = []

    # Sort data by priority
    for key in priority_order:
        if key in data:
            prioritized_data.append((key, data[key]))

    # Calculate total size and trim data to fit bandwidth
    total_size = sum(len(item[1]) for item in prioritized_data)
    if total_size > bandwidth_limit:
        for i, (key, value) in enumerate(prioritized_data):
            allowed_size = int(len(value) * (bandwidth_limit / total_size))
            prioritized_data[i] = (key, value[:allowed_size])

    return dict(prioritized_data)