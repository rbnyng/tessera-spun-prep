"""
UK National Parks Time Series Inference

Generates mycorrhizal richness prediction TIFs for UK national parks
using Tessera embeddings and a pre-trained SSL model.

Parks: Cairngorms, Lake District, Yorkshire Dales
Years: 2017, 2024 (extensible to full 2017-2024 range)

Usage:
    python uk_national_parks_inference.py
"""

import subprocess
import pickle
import json
import zipfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from rasterio.merge import merge
import rasterio
from rasterio.transform import from_origin
from skimage.util import view_as_windows
import geopandas as gpd


class Config:
    # === Training/Model Paths (existing infrastructure) ===
    TRAINING_BIODIVERSITY_CSVS = [
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_europe.csv",
        "/maps-priv/maps/ray25/data/spun_data/ECM_richness_Asia.csv"
    ]
    TRAINING_REPRESENTATIONS_DIR = "/maps-priv/maps/ray25/data/ecm_representations"
    MODEL_OUTPUT_DIR = Path("./model")
    EVALUATOR_SAVE_PATH = MODEL_OUTPUT_DIR / "evaluator_ssl_only.pkl"
    MODEL_SSL_ONLY_SAVE_PATH = MODEL_OUTPUT_DIR / "model_ssl_only.pkl"
    SATELLITE_DIM_REDUCTION = 'umap'
    DIM_REDUCTION_COMPONENTS = 256
    USE_BIOME_FILTER = True

    # === UK National Parks Shapefiles ===
    NATIONAL_PARKS_DIR = Path("/maps-priv/maps/ray25/nationalparks")

    # Map of park names to their shapefile zip paths
    # Adjust these filenames to match your actual files
    PARK_SHAPEFILES = {
        "cairngorms": NATIONAL_PARKS_DIR / "cairngorms.zip",
        "lake_district": NATIONAL_PARKS_DIR / "lake_district.zip",
        "yorkshire_dales": NATIONAL_PARKS_DIR / "yorkshire_dales.zip",
    }

    # === Inference Configuration ===
    YEARS = [2017, 2024]  # Can extend to list(range(2017, 2025)) for all years

    # === Output Paths ===
    OUTPUT_BASE_DIR = Path("./data/uk_national_parks")

    @classmethod
    def get_park_output_dir(cls, park_name: str) -> Path:
        return cls.OUTPUT_BASE_DIR / park_name

    @classmethod
    def get_embeddings_dir(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"embeddings_{year}"

    @classmethod
    def get_mosaic_path(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"mosaic_{year}.tif"

    @classmethod
    def get_prediction_path(cls, park_name: str, year: int) -> Path:
        return cls.get_park_output_dir(park_name) / f"prediction_{year}.tif"

    @classmethod
    def get_aoi_geojson_path(cls, park_name: str) -> Path:
        return cls.get_park_output_dir(park_name) / "aoi.geojson"


def load_park_boundary(shapefile_path: Path) -> gpd.GeoDataFrame:
    """
    Load national park boundary from a zipped shapefile.

    Handles both:
    - Direct .zip files containing shapefiles
    - Directories containing .shp files
    """
    if shapefile_path.suffix == '.zip':
        # Read directly from zip
        gdf = gpd.read_file(f"zip://{shapefile_path}")
    elif shapefile_path.suffix == '.shp':
        gdf = gpd.read_file(shapefile_path)
    elif shapefile_path.is_dir():
        # Find .shp file in directory
        shp_files = list(shapefile_path.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp file found in {shapefile_path}")
        gdf = gpd.read_file(shp_files[0])
    else:
        raise ValueError(f"Unsupported shapefile format: {shapefile_path}")

    # Ensure WGS84 for geotessera
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def gdf_to_bbox_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    Convert a GeoDataFrame to a bounding box GeoJSON FeatureCollection.

    Uses the total bounds of all geometries to create a simple bbox polygon.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy],
                    [minx, miny]
                ]]
            }
        }]
    }


def ensure_model_exists(config: Config):
    """
    Check that the pre-trained SSL model exists.
    If not, provide instructions for training.
    """
    if not config.MODEL_SSL_ONLY_SAVE_PATH.exists():
        print("ERROR: Pre-trained SSL model not found!")
        print(f"Expected path: {config.MODEL_SSL_ONLY_SAVE_PATH}")
        print("\nPlease run the training step first using 10m_map.py or time_series_diff.py")
        print("Or copy the existing model files to the ./model/ directory")
        exit(1)

    if not config.EVALUATOR_SAVE_PATH.exists():
        print("ERROR: Evaluator object not found!")
        print(f"Expected path: {config.EVALUATOR_SAVE_PATH}")
        exit(1)

    print("✓ Pre-trained model and evaluator found")


def download_embeddings_for_park(config: Config, park_name: str, year: int, aoi_geojson_path: Path):
    """
    Download Tessera embeddings for a given park and year.
    """
    embeddings_dir = config.get_embeddings_dir(park_name, year)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # geotessera nests files: global_0.1_degree_representation/{year}/grid_*/grid_*_{year}.tiff
    repr_dir = embeddings_dir / 'global_0.1_degree_representation' / str(year)
    if repr_dir.exists() and any(repr_dir.glob('*/*.tiff')):
        print(f"  Embeddings for {park_name} {year} already exist. Skipping download.")
        return

    print(f"  Downloading Tessera embeddings for {park_name} {year}...")
    command = [
        'geotessera', 'download',
        '--region-file', str(aoi_geojson_path),
        '--year', str(year),
        '--format', 'tiff',
        '--output', str(embeddings_dir)
    ]

    try:
        subprocess.run(command, check=True)
        print(f"  Download complete for {park_name} {year}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to download embeddings for {park_name} {year}")
        print(f"  Command: {' '.join(command)}")
        raise


def run_inference_per_tile(config: Config, park_name: str, year: int, evaluator, model):
    """
    Run SSL model inference on each embedding tile individually.
    Saves per-tile predictions to a subdirectory.
    """
    embeddings_dir = config.get_embeddings_dir(park_name, year)
    predictions_dir = config.get_park_output_dir(park_name) / f"predictions_tiles_{year}"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Find embedding tiles
    repr_dir = embeddings_dir / 'global_0.1_degree_representation' / str(year)
    embedding_files = list(repr_dir.glob('*/*.tiff')) + list(repr_dir.glob('*/*.tif'))

    if not embedding_files:
        print(f"  ERROR: No embedding tiles found for {park_name} {year}")
        return False

    print(f"  Running inference on {len(embedding_files)} tiles...")

    processed = 0
    skipped = 0

    for i, tile_path in enumerate(embedding_files):
        # Output prediction tile path
        pred_tile_path = predictions_dir / f"{tile_path.stem}_pred.tif"

        if pred_tile_path.exists():
            skipped += 1
            continue

        try:
            with rasterio.open(tile_path) as src:
                tile_data = src.read().transpose(1, 2, 0)  # (H, W, C)
                profile = src.profile

            h, w, c = tile_data.shape

            # Create 3x3 windows for spatial context
            padded = np.pad(tile_data, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
            windows = view_as_windows(padded, (3, 3, c), step=1)
            feature_vectors = windows.reshape(h * w, -1)

            # Filter valid pixels
            valid_mask = ~np.isnan(feature_vectors).any(axis=1)
            valid_features = feature_vectors[valid_mask]

            if valid_features.shape[0] == 0:
                # All NaN tile - skip
                continue

            # Apply model pipeline
            scaled_pixels = evaluator.scaler.transform(valid_features)
            reduced_pixels = evaluator.dim_reduction_model.transform(scaled_pixels)
            predictions_flat = model.predict(reduced_pixels)

            # Reconstruct prediction map
            final_predictions = np.full(h * w, np.nan, dtype=np.float32)
            final_predictions[valid_mask] = predictions_flat
            prediction_map = final_predictions.reshape(h, w)

            # Save prediction tile
            profile.update(count=1, dtype='float32', nodata=np.nan, compress='LZW')
            with rasterio.open(pred_tile_path, 'w', **profile) as dst:
                dst.write(prediction_map, 1)

            processed += 1

            # Progress update every 10 tiles
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(embedding_files)} tiles...")

        except Exception as e:
            print(f"    ERROR processing {tile_path.name}: {e}")
            continue

    print(f"  Inference complete: {processed} new, {skipped} skipped (already exist)")
    return True


def merge_prediction_tiles(config: Config, park_name: str, year: int):
    """
    Merge per-tile predictions into a single prediction GeoTIFF.
    Only merges single-band predictions (much lower memory than 128-band embeddings).
    """
    predictions_dir = config.get_park_output_dir(park_name) / f"predictions_tiles_{year}"
    final_prediction_path = config.get_prediction_path(park_name, year)

    if final_prediction_path.exists():
        print(f"  Final prediction for {park_name} {year} already exists. Skipping merge.")
        return

    pred_tiles = list(predictions_dir.glob('*_pred.tif'))
    if not pred_tiles:
        print(f"  ERROR: No prediction tiles found to merge for {park_name} {year}")
        return

    print(f"  Merging {len(pred_tiles)} prediction tiles...")

    # Open all prediction tiles
    src_files = [rasterio.open(fp) for fp in pred_tiles]
    mosaic_data, out_transform = merge(src_files)

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic_data.shape[1],
        "width": mosaic_data.shape[2],
        "transform": out_transform,
        "compress": "LZW",
        "BIGTIFF": "IF_SAFER"
    })

    with rasterio.open(final_prediction_path, "w", **out_meta) as dest:
        dest.write(mosaic_data)

    for src in src_files:
        src.close()

    print(f"  Final prediction saved: {final_prediction_path}")

    # Print stats
    valid_preds = mosaic_data[~np.isnan(mosaic_data)]
    if valid_preds.size > 0:
        print(f"    Richness range: {valid_preds.min():.1f} - {valid_preds.max():.1f}")
        print(f"    Richness mean: {valid_preds.mean():.1f}")


def process_park(config: Config, park_name: str, evaluator, model):
    """
    Full processing pipeline for a single national park.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {park_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")

    # Setup output directory
    output_dir = config.get_park_output_dir(park_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load park boundary
    shapefile_path = config.PARK_SHAPEFILES[park_name]
    print(f"\nLoading boundary from: {shapefile_path}")

    try:
        gdf = load_park_boundary(shapefile_path)
        print(f"  Loaded {len(gdf)} feature(s)")
        print(f"  Total bounds: {gdf.total_bounds}")
    except Exception as e:
        print(f"  ERROR loading shapefile: {e}")
        return

    # Create AOI GeoJSON (bounding box)
    aoi_geojson = gdf_to_bbox_geojson(gdf)
    aoi_path = config.get_aoi_geojson_path(park_name)
    with open(aoi_path, 'w') as f:
        json.dump(aoi_geojson, f)
    print(f"  AOI GeoJSON saved: {aoi_path}")

    # Process each year
    for year in config.YEARS:
        print(f"\n--- Year {year} ---")
        download_embeddings_for_park(config, park_name, year, aoi_path)
        if run_inference_per_tile(config, park_name, year, evaluator, model):
            merge_prediction_tiles(config, park_name, year)


def main():
    print("="*60)
    print("UK National Parks - Mycorrhizal Richness Prediction")
    print("="*60)

    config = Config()

    # Verify model exists
    print("\nChecking for pre-trained model...")
    ensure_model_exists(config)

    # Load model and evaluator once
    print("\nLoading model and evaluator...")
    with open(config.EVALUATOR_SAVE_PATH, 'rb') as f:
        evaluator = pickle.load(f)
    with open(config.MODEL_SSL_ONLY_SAVE_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")

    # Process each park
    parks_to_process = list(config.PARK_SHAPEFILES.keys())
    print(f"\nParks to process: {parks_to_process}")
    print(f"Years: {config.YEARS}")

    for park_name in parks_to_process:
        try:
            process_park(config, park_name, evaluator, model)
        except Exception as e:
            print(f"\nERROR processing {park_name}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {config.OUTPUT_BASE_DIR}")
    print("\nGenerated files:")

    for park_name in parks_to_process:
        park_dir = config.get_park_output_dir(park_name)
        if park_dir.exists():
            print(f"\n{park_name}/")
            for f in sorted(park_dir.glob("prediction_*.tif")):
                print(f"  {f.name}")


if __name__ == "__main__":
    main()
