# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:26:30 2026

@author: danielsilva
"""

# %% --- 1. Prepare Training Data (Convert XML to CSV) ---
import os
import pandas as pd
import glob
from deepforest import utilities
import xml.etree.ElementTree as ET

# --- SET THESE PATHS ---
export_dir = "Replace with export directory path"
xml_dir = os.path.join(export_dir, "Annotations")
img_dir = os.path.join(export_dir, "images")
output_csv = os.path.join(export_dir, "train_annotations.csv")

print("Converting Pascal VOC XMLs to DeepForest CSV format...")

records = []
xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract filename tag or fallback to the XML basename
        filename_node = root.find("filename")
        if filename_node is not None and filename_node.text:
            image_name = filename_node.text
        else:
            image_name = os.path.basename(xml_file).replace(".xml", ".jpg")

        # Keep only the image filename (basename), not full folder paths
        image_name = os.path.basename(image_name)

        # Parse each bounding box inside the XML
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                records.append({
                    "image_path": image_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": "Tree"  # DeepForest requires the standard 'Tree' label
                })

    except Exception as e:
        print(f"Failed to parse {os.path.basename(xml_file)}: {e}")

# Save output DataFrame
if records:
    train_df = pd.DataFrame(records)
    train_df.to_csv(output_csv, index=False)
    print(f"\nData successfully prepared!")
    print(f"Total bounding boxes extracted: {len(train_df)}")
    print(f"Unique images annotated: {train_df['image_path'].nunique()}")
    print(f"CSV saved to: {output_csv}")
else:
    raise ValueError("No bounding boxes were extracted. Please verify the folder contents.")
    
    
    
# %% --- 2. Fine-Tune DeepForest on GPU (NVIDIA RTX 2000 Ada) ---
import os
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
from deepforest import main

# Prevent OpenMP library duplicate conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------------------
# STEP 1: Verify CUDA / GPU Availability
# -------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in PyTorch! Check that your NVIDIA driver and CUDA PyTorch are installed."
    )

gpu_name = torch.cuda.get_device_name(0)
print(f"CUDA detected! Training on GPU: {gpu_name}")

# -------------------------------------------------------------
# STEP 2: Initialize Baseline Model
# -------------------------------------------------------------
model = main.deepforest()
#model.use_release()
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# -------------------------------------------------------------
# STEP 3: Configure Training & GPU Parameters
# -------------------------------------------------------------
# File paths
model.config["train"]["csv_file"] = output_csv
model.config["train"]["root_dir"] = img_dir

# GPU Acceleration
model.config["accelerator"] = "gpu"
model.config["devices"] = 1

# Hyperparameters tuned for RTX 2000 Ada VRAM
model.config["batch_size"] = 6        # Increased from 2 to 6 (RTX 2000 Ada can handle 6-8 images per batch)
model.config["workers"] = 4           # Asynchronous CPU threads for feeding images to the GPU
model.config["train"]["epochs"] = 15   # 15 epochs is standard for transfer learning
model.config["train"]["lr"] = 0.001    # Learning rate

# -------------------------------------------------------------
# STEP 4: Build Trainer and Start Fine-Tuning
# -------------------------------------------------------------
print("Building PyTorch Lightning Trainer...")
model.create_trainer()

print("Starting GPU training run...")
model.trainer.fit(model)

print("Fine-tuning complete!")


# %% --- 3. Save Fine-Tuned Model Weights ---
save_path = "Replace with path to model output directory/erva_mate_model.pt"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Save weights
torch.save(model.model.state_dict(), save_path)
print(f"Custom model weights successfully saved to:\n -> {save_path}")