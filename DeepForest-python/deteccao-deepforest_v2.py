# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:21:16 2026

@author: danielsilva
"""

# %% --- Full Erva-Mate Tree Detection Pipeline (Alpha-Stripped) ---
import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import torchvision
from shapely.geometry import box
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from deepforest import main


# %% --- 1. Environment Safeguards
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)


# %% --- 2. File Paths Configuration
base_path = "Replace with your base directory path"  # Or your local un-synced folder
model_path = "Replace with your model path"  #modelo 100%
raster_path = os.path.join(base_path, "orto.tif")
chm_path = os.path.join(base_path, "CHM.tif")
output_shp_path = os.path.join(base_path, "arvs_deepforest.shp")
output_labels_path = os.path.join(base_path, "tree_labels.tif")


# %% --- 3. Initialize Model
print("Loading DeepForest model onto CUDA...", flush=True)
model = main.deepforest()
model.config["workers"] = 0
model.model.load_state_dict(
    torch.load(model_path, map_location=model.device, weights_only=True)
)
#model.load_model(model_name="weecology/deepforest-tree", revision="main")

#model.to("cuda")
model.model.eval()


# %% --- 4. Sliding Window Detection (Stripping Alpha Channel)
patch_size = 1000
step_size = 750  # 25% overlap
all_raw_boxes = []

print("Running sliding window predictions across raster...", flush=True)
with rasterio.open(raster_path) as src:
    width, height = src.width, src.height
    crs = src.crs
    transform = src.transform

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            w_w = min(patch_size, width - x)
            w_h = min(patch_size, height - y)

            if w_w < 200 or w_h < 200:
                continue

            window = Window(col_off=x, row_off=y, width=w_w, height=w_h)
            
            # CRITICAL FIX: Explicitly read ONLY Bands 1, 2, 3 (Ignores Band 4 Alpha)
            tile_data = src.read((1, 2, 3), window=window)

            # Skip pure black background or no-data tiles
            if np.all(tile_data == 0) or np.std(tile_data) < 2.0:
                continue

            # Format to (H, W, C) uint8 and enforce memory contiguity
            tile_img = np.transpose(tile_data, (1, 2, 0))
            if tile_img.dtype != np.uint8:
                tile_img = (tile_img / tile_img.max() * 255).astype(np.uint8)
            tile_img = np.ascontiguousarray(tile_img)

            # Predict on single 3-channel RGB tile
            boxes = model.predict_image(image=tile_img)

            if boxes is not None and not boxes.empty:
                boxes = boxes.dropna(subset=["xmin", "ymin", "xmax", "ymax", "score"])
                if not boxes.empty:
                    # Convert tile-relative coordinates to global pixel coordinates
                    boxes["xmin"] += x
                    boxes["xmax"] += x
                    boxes["ymin"] += y
                    boxes["ymax"] += y
                    all_raw_boxes.append(boxes)

            torch.cuda.empty_cache()
            
# Combine all tile detections
raw_df = pd.concat(all_raw_boxes, ignore_index=True)
print(f"Total raw bounding boxes detected: {len(raw_df)}", flush=True)


# %% ---  5. Global Non-Maximum Suppression (NMS) to Deduplicate Overlapping Tiles
print("Applying Global NMS to remove overlapping window detections...", flush=True)
boxes_tensor = torch.tensor(raw_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
scores_tensor = torch.tensor(raw_df["score"].values, dtype=torch.float32)

keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.25)
dedup_df = raw_df.iloc[keep_indices.numpy()].copy().reset_index(drop=True)

print(f"Detections remaining after NMS deduplication: {len(dedup_df)}", flush=True)           



# %% --- 6. Filter Detections by Size and Score
with rasterio.open(raster_path) as src:
    pixel_res = abs(src.transform[0])

dedup_df['width_px'] = dedup_df['xmax'] - dedup_df['xmin']
dedup_df['height_px'] = dedup_df['ymax'] - dedup_df['ymin']

min_px = 0.6 / pixel_res
max_px = 6.0 / pixel_res

score_mask = dedup_df['score'] > 0.1
width_mask = (dedup_df['width_px'] >= min_px) & (dedup_df['width_px'] <= max_px)
height_mask = (dedup_df['height_px'] >= min_px) & (dedup_df['height_px'] <= max_px)

filtered_df = dedup_df[score_mask & width_mask & height_mask].copy()

# 7. Canopy Height Model (CHM) Filtering
print("Filtering detections using Canopy Height Model (CHM)...", flush=True)
with rasterio.open(chm_path) as chm_src:
    chm_data = chm_src.read(1)
    nodata_val = chm_src.nodata
    if nodata_val is not None:
        chm_data = np.where(chm_data == nodata_val, np.nan, chm_data)

chm_heights = []
for _, row in filtered_df.iterrows():
    xmin, ymin = max(0, int(row['xmin'])), max(0, int(row['ymin']))
    xmax, ymax = min(chm_data.shape[1], int(row['xmax'])), min(chm_data.shape[0], int(row['ymax']))
    box_chm = chm_data[ymin:ymax, xmin:xmax]
    chm_heights.append(np.nanmax(box_chm) if box_chm.size > 0 and not np.all(np.isnan(box_chm)) else np.nan)

filtered_df['chm_height_m'] = chm_heights
chm_mask = (filtered_df['chm_height_m'] >= 0.5) & (filtered_df['chm_height_m'] <= 10.0)
chm_filtered_df = filtered_df[chm_mask].copy()

print(f"Final valid trees detected (CHM 0.5m-10.0m): {len(chm_filtered_df)}", flush=True)


# %% --- 8. Export Shapefile
geometries = []
for _, row in chm_filtered_df.iterrows():
    x_min_map, y_top_map = transform * (row['xmin'], row['ymin'])
    x_max_map, y_bottom_map = transform * (row['xmax'], row['ymax'])
    geometries.append(box(min(x_min_map, x_max_map), min(y_top_map, y_bottom_map), max(x_min_map, x_max_map), max(y_top_map, y_bottom_map)))

gdf = gpd.GeoDataFrame(chm_filtered_df, geometry=geometries, crs=crs)
gdf.to_file(output_shp_path, driver="ESRI Shapefile")
print(f"Exported final shapefile to: {output_shp_path}", flush=True)



# %% --- Cleanup Memory
del model
torch.cuda.empty_cache()
gc.collect()

































