import torch
import h5py
import os
import numpy as np
from pathlib import Path, WindowsPath
import pickle
import traceback
from glob import glob
from natsort import natsorted
from tqdm import tqdm

class ModelConverter:
    def __init__(self):
        self.models_dir = os.path.abspath('models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def convert_pkl_to_h5(self):
        """Convert FastAI pkl model to H5 format"""
        try:
            # Get all pkl files in models directory
            pkl_files = natsorted(glob(os.path.join(self.models_dir, '**/*.pkl'), recursive=True))
            print(f"Found {len(pkl_files)} pkl files in {self.models_dir}")

            # Output h5 file path
            h5_path = os.path.join(self.models_dir, 'wound_model.h5')
            print(f"Writing output to {h5_path}")

            # Create custom unpickler to handle persistent_load
            class CustomUnpickler(pickle.Unpickler):
                def persistent_load(self, pid):
                    return pid

                def find_class(self, module, name):
                    if module == 'pathlib':
                        return WindowsPath
                    return super().find_class(module, name)

            # Create H5 file
            with h5py.File(h5_path, 'w') as hf:
                # Create groups for model data
                hf_params = hf.create_group('model_params')
                hf_categories = hf.create_group('categories')
                hf_metadata = hf.create_group('metadata')

                # Track unique dataset names
                dataset_names = set()

                # Process each pkl file
                for pkl_file in tqdm(pkl_files, desc="Converting..."):
                    # Get dataset name from filename
                    dataset_name = os.path.splitext(os.path.basename(pkl_file))[0]
                    if dataset_name in dataset_names:
                        print(f"Warning: Filename {dataset_name} appears more than once")
                        continue
                    dataset_names.add(dataset_name)

                    # Load the pkl file
                    try:
                        with open(pkl_file, 'rb') as f:
                            # Use custom unpickler
                            unpickler = CustomUnpickler(f)
                            try:
                                model_data = unpickler.load()
                            except Exception as pickle_error:
                                print(f"Pickle load failed, trying torch.load: {pickle_error}")
                                f.seek(0)
                                model_data = torch.load(
                                    f,
                                    map_location=torch.device('cpu'),
                                    pickle_module=pickle
                                )

                        # Extract model parameters
                        if isinstance(model_data, dict):
                            # If it's a dictionary, save directly
                            for key, value in model_data.items():
                                if isinstance(value, np.ndarray):
                                    hf_params.create_dataset(f"{dataset_name}/{key}", data=value)
                        else:
                            # If it's a model, try to get state dict
                            try:
                                if hasattr(model_data, 'model'):
                                    state_dict = model_data.model.state_dict()
                                elif hasattr(model_data, 'state_dict'):
                                    state_dict = model_data.state_dict()
                                else:
                                    print(f"Warning: Could not find state dict in {pkl_file}")
                                    continue
                                
                                for name, param in state_dict.items():
                                    param_data = param.cpu().numpy()
                                    hf_params.create_dataset(f"{dataset_name}/{name}", data=param_data)

                                # Try to get categories if available
                                if hasattr(model_data, 'dls') and hasattr(model_data.dls, 'vocab'):
                                    categories = model_data.dls.vocab
                                    cat_group = hf_categories.create_group(dataset_name)
                                    for i, cat in enumerate(categories):
                                        cat_group.attrs[str(i)] = str(cat)

                            except Exception as e:
                                print(f"Warning: Could not process model in {pkl_file}: {e}")
                                continue

                    except Exception as e:
                        print(f"Error processing {pkl_file}: {e}")
                        print("Detailed error:")
                        traceback.print_exc()
                        continue

                # Save metadata
                hf_metadata.attrs['conversion_date'] = str(np.datetime64('now'))
                hf_metadata.attrs['num_models'] = len(dataset_names)
                hf_metadata.attrs['model_names'] = ','.join(sorted(dataset_names))

            print(f"\nConversion completed! Saved to: {h5_path}")
            return True

        except Exception as e:
            print(f"Error in conversion process: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()
            return False

    def verify_h5_model(self):
        """Verify the converted H5 model"""
        try:
            h5_path = os.path.join(self.models_dir, 'wound_model.h5')
            
            if not os.path.exists(h5_path):
                print("H5 model file not found!")
                return
            
            print(f"\nVerifying H5 model at: {h5_path}")
            
            with h5py.File(h5_path, 'r') as f:
                print("\nModel Structure:")
                
                def print_group(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"Group: {name}")
                        if 'categories' in name:
                            print("Categories:")
                            for key, value in obj.attrs.items():
                                print(f"  {key}: {value}")
                
                f.visititems(print_group)
                
                print("\nMetadata:")
                if 'metadata' in f:
                    for key, value in f['metadata'].attrs.items():
                        print(f"  {key}: {value}")

            print("\nH5 model verified successfully!")
            
        except Exception as e:
            print(f"Error verifying H5 model: {str(e)}")
            print("Detailed error:")
            traceback.print_exc()

def main():
    converter = ModelConverter()
    if converter.convert_pkl_to_h5():
        converter.verify_h5_model()

if __name__ == "__main__":
    main()