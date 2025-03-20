import os
import cv2
import numpy as np
import h5py
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

class WoundDatasetConverter:
    def __init__(self):
        self.datasets_dir = Path('datasets')
        self.models_dir = Path('models')
        self.image_size = (224, 224)
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.image_size)
            
            # Normalize pixel values
            img = img.astype('float32') / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def convert_dataset(self):
        """Convert wound dataset to H5 format"""
        try:
            # Get wound dataset path
            injury_path = self.datasets_dir / 'injuries' / 'Wound_dataset'
            if not injury_path.exists():
                raise FileNotFoundError(f"Wound dataset folder not found at {injury_path}")

            # Get wound types from subdirectories
            wound_types = [d for d in os.listdir(injury_path) 
                         if os.path.isdir(os.path.join(injury_path, d))]
            
            if not wound_types:
                raise ValueError(f"No wound type folders found in {injury_path}")
            
            print(f"Found wound types: {wound_types}")
            
            # Create label mapping
            label_mapping = {label: idx for idx, label in enumerate(sorted(wound_types))}
            
            # Initialize lists for data and labels
            images = []
            labels = []
            
            # Process each wound type
            for wound_type in tqdm(wound_types, desc="Processing wound types"):
                wound_dir = injury_path / wound_type
                
                # Get all image files
                image_files = [f for f in wound_dir.glob('*') 
                             if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                print(f"\nProcessing {wound_type}: {len(image_files)} images")
                
                # Process each image
                for img_path in tqdm(image_files, desc=f"Processing {wound_type}"):
                    img = self.load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(label_mapping[wound_type])
            
            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)
            
            # Create H5 file
            h5_path = self.models_dir / 'wound_dataset.h5'
            with h5py.File(h5_path, 'w') as f:
                # Store data
                f.create_dataset('images', data=X)
                f.create_dataset('labels', data=y)
                
                # Store metadata
                f.attrs['image_size'] = self.image_size
                f.attrs['num_classes'] = len(wound_types)
                
                # Store label mapping
                label_group = f.create_group('label_mapping')
                for label, idx in label_mapping.items():
                    label_group.attrs[label] = idx
            
            print(f"\nDataset converted successfully!")
            print(f"Total images: {len(images)}")
            print(f"Dataset saved to: {h5_path}")
            print(f"Image size: {self.image_size}")
            print(f"Number of classes: {len(wound_types)}")
            print("Label mapping:")
            for label, idx in label_mapping.items():
                print(f"  {label}: {idx}")
                
        except Exception as e:
            print(f"Error converting dataset: {str(e)}")

    def verify_h5_file(self):
        """Verify the created H5 file"""
        try:
            h5_path = self.models_dir / 'wound_dataset.h5'
            if not h5_path.exists():
                print("H5 file not found!")
                return
                
            with h5py.File(h5_path, 'r') as f:
                # Print dataset info
                print("\nH5 File Verification:")
                print("Datasets:")
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        print(f"  {key}: shape={f[key].shape}, dtype={f[key].dtype}")
                    elif isinstance(f[key], h5py.Group):
                        print(f"  {key}: (Group)")
                
                # Print attributes
                print("\nAttributes:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                
                # Print label mapping
                if 'label_mapping' in f:
                    print("\nLabel Mapping:")
                    for key, value in f['label_mapping'].attrs.items():
                        print(f"  {key}: {value}")
                
            print("\nH5 file verified successfully!")
            
        except Exception as e:
            print(f"Error verifying H5 file: {str(e)}")

def main():
    converter = WoundDatasetConverter()
    converter.convert_dataset()
    converter.verify_h5_file()

if __name__ == "__main__":
    main() 