from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod, Transformer
from global_land_mask import globe
import torch

from config import (
    BEST_MODEL_PATH,
    DEVICE,
    MASK_THRESHOLD,
    UNET_BASE_CHANNELS,
)

from src.models.unet import UNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_TIF = (
    BASE_DIR
    / "data"
    / "sentinel1"
    / "processed"
    / "sentinel1_gulf_vv_geocoded.tif"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "sentinel1"
    / "processed"
    / "latest_scene.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "sentinel1_real"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 256

MIN_COMPONENT_AREA = 50

# Stronger coastline exclusion
COAST_BUFFER_PIXELS = 12

GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# METADATA
# ============================================================

def load_scene_metadata():

    if not METADATA_FILE.exists():

        return {
            "source": "Copernicus Sentinel-1",
            "source_product": None,
            "acquisition_timestamp": None,
            "platform": None,
            "polarization": ["VV"],
            "instrument_mode": "IW"
        }

    return json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# LAND MASK
# ============================================================

def create_land_mask(
    height,
    width,
    transform,
    crs
):

    print("Creating geographic land mask...")

    rows, cols = np.indices(
        (
            height,
            width
        ),
        dtype=np.float64
    )

    xs = (
        transform.c
        + transform.a * (
            cols + 0.5
        )
        + transform.b * (
            rows + 0.5
        )
    )

    ys = (
        transform.f
        + transform.d * (
            cols + 0.5
        )
        + transform.e * (
            rows + 0.5
        )
    )

    if (
        crs is not None
        and str(crs) != "EPSG:4326"
    ):

        transformer = Transformer.from_crs(
            crs,
            "EPSG:4326",
            always_xy=True
        )

        lon, lat = transformer.transform(
            xs,
            ys
        )

    else:

        lon = xs
        lat = ys

    land_mask = globe.is_land(
        lat,
        lon
    )

    return np.asarray(
        land_mask,
        dtype=bool
    )


# ============================================================
# COAST BUFFER
# ============================================================

def create_buffered_land_mask(
    land_mask,
    buffer_pixels
):

    land_uint8 = (
        land_mask.astype(
            np.uint8
        )
    )

    if buffer_pixels <= 0:
        return land_mask

    size = (
        buffer_pixels * 2
        + 1
    )

    kernel = np.ones(
        (
            size,
            size
        ),
        dtype=np.uint8
    )

    buffered = cv2.dilate(
        land_uint8,
        kernel,
        iterations=1
    )

    return (
        buffered > 0
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sar(
    image,
    valid_ocean_mask
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    usable = (
        valid_ocean_mask
        & np.isfinite(
            image
        )
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    if not usable.any():
        return output

    values = image[
        usable
    ]

    low = np.percentile(
        values,
        1
    )

    high = np.percentile(
        values,
        99
    )

    if high <= low:
        return output

    clipped = np.clip(
        image,
        low,
        high
    )

    output[
        usable
    ] = (
        clipped[usable]
        - low
    ) / (
        high
        - low
    )

    output[
        ~usable
    ] = 0

    return output


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=UNET_BASE_CHANNELS
    ).to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ============================================================
# TILE PREDICTION
# ============================================================

def predict_tile(
    model,
    tile
):

    tensor = (
        torch
        .from_numpy(tile)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = (
            torch
            .sigmoid(
                logits
            )[0, 0]
            .cpu()
            .numpy()
        )

    return probability


# ============================================================
# COMPONENT CLEANING
# ============================================================

def clean_mask(
    mask,
    ocean_mask,
    min_area=50
):

    mask = mask.astype(
        np.uint8
    )

    ocean_mask = ocean_mask.astype(
        bool
    )

    # Hard rule: no prediction outside ocean
    mask[
        ~ocean_mask
    ] = 0

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    h, w = mask.shape

    # Create a safety band around invalid/land pixels
    invalid = (
        ~ocean_mask
    ).astype(
        np.uint8
    )

    invalid_buffer = cv2.dilate(
        invalid,
        np.ones(
            (7, 7),
            np.uint8
        ),
        iterations=1
    ).astype(
        bool
    )

    for label in range(
        1,
        number_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        component = (
            labels == label
        )

        touches_border = (
            component[0, :].any()
            or component[h - 1, :].any()
            or component[:, 0].any()
            or component[:, w - 1].any()
        )

        touches_land_or_coast = (
            component
            & invalid_buffer
        ).any()

        if (
            touches_border
            or touches_land_or_coast
        ):
            continue

        cleaned[
            component
        ] = 1

    # Final hard mask AGAIN
    cleaned[
        ~ocean_mask
    ] = 0

    return cleaned


# ============================================================
# AREA
# ============================================================

def calculate_area_km2(
    geometry
):

    area_m2, _ = (
        GEOD.geometry_area_perimeter(
            geometry
        )
    )

    return (
        abs(
            area_m2
        )
        / 1_000_000
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "OILTRACE AI - SENTINEL-1 OCEAN-ONLY INFERENCE"
    )
    print("=" * 72)

    scene_metadata = (
        load_scene_metadata()
    )

    print(
        "Scene:",
        scene_metadata.get(
            "source_product"
        )
    )

    print(
        "Acquisition:",
        scene_metadata.get(
            "acquisition_timestamp"
        )
    )

    if not INPUT_TIF.exists():

        raise FileNotFoundError(
            f"Input GeoTIFF not found:\n{INPUT_TIF}"
        )

    model = load_model()

    print(
        "✓ U-Net model loaded"
    )

    with rasterio.open(
        INPUT_TIF
    ) as src:

        vv = src.read(
            1
        )

        data_mask = src.read(
            2
        )

        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        bounds = src.bounds

    print(
        "✓ Sentinel-1 GeoTIFF loaded"
    )

    print(
        "Size:",
        width,
        "x",
        height
    )

    print(
        "CRS:",
        crs
    )

    print(
        "Bounds:",
        bounds
    )

    # ========================================================
    # VALID SATELLITE COVERAGE
    # ========================================================

    satellite_valid_mask = (
        data_mask > 0
    )

    print(
        "Satellite valid pixels:",
        int(
            satellite_valid_mask.sum()
        )
    )

    # ========================================================
    # LAND MASK
    # ========================================================

    land_mask = create_land_mask(
        height,
        width,
        transform,
        crs
    )

    print(
        "✓ Land mask created"
    )

    print(
        "Land pixels:",
        int(
            land_mask.sum()
        )
    )

    # ========================================================
    # BUFFER COAST
    # ========================================================

    buffered_land_mask = (
        create_buffered_land_mask(
            land_mask,
            COAST_BUFFER_PIXELS
        )
    )

    print(
        f"✓ Coast buffer applied: "
        f"{COAST_BUFFER_PIXELS} pixels"
    )

    # ========================================================
    # FINAL USABLE OCEAN MASK
    # ========================================================

    ocean_mask = (
        satellite_valid_mask
        & ~buffered_land_mask
    )

    ocean_pixels = int(
        ocean_mask.sum()
    )

    print(
        "Usable ocean pixels:",
        ocean_pixels
    )

    if ocean_pixels == 0:

        raise RuntimeError(
            "No valid ocean pixels found in selected AOI."
        )

    # ========================================================
    # NORMALIZE OCEAN ONLY
    # ========================================================

    vv = normalize_sar(
        vv,
        ocean_mask
    )

    print(
        "✓ Ocean-only normalization complete"
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    final_probability = np.zeros(
        (
            height,
            width
        ),
        dtype=np.float32
    )

    total_tiles = (
        int(
            np.ceil(
                height / TILE_SIZE
            )
        )
        *
        int(
            np.ceil(
                width / TILE_SIZE
            )
        )
    )

    tile_counter = 0
    model_tiles = 0

    for y in range(
        0,
        height,
        TILE_SIZE
    ):

        for x in range(
            0,
            width,
            TILE_SIZE
        ):

            tile_counter += 1

            valid_h = min(
                TILE_SIZE,
                height - y
            )

            valid_w = min(
                TILE_SIZE,
                width - x
            )

            tile = vv[
                y:y + valid_h,
                x:x + valid_w
            ]

            tile_ocean = ocean_mask[
                y:y + valid_h,
                x:x + valid_w
            ]

            if not tile_ocean.any():

                print(
                    f"\rTiles: "
                    f"{tile_counter}/"
                    f"{total_tiles}",
                    end=""
                )

                continue

            padded = np.zeros(
                (
                    TILE_SIZE,
                    TILE_SIZE
                ),
                dtype=np.float32
            )

            padded[
                :valid_h,
                :valid_w
            ] = tile

            probability = predict_tile(
                model,
                padded
            )

            probability = probability[
                :valid_h,
                :valid_w
            ]

            # Strict ocean-only probability
            probability[
                ~tile_ocean
            ] = 0

            final_probability[
                y:y + valid_h,
                x:x + valid_w
            ] = probability

            model_tiles += 1

            print(
                f"\rTiles: "
                f"{tile_counter}/"
                f"{total_tiles}",
                end=""
            )

    print()

    print(
        "✓ Tile-wise inference complete"
    )

    print(
        "Tiles processed:",
        model_tiles
    )

    # ========================================================
    # FINAL HARD MASK BEFORE THRESHOLD
    # ========================================================

    final_probability[
        ~ocean_mask
    ] = 0

    # ========================================================
    # THRESHOLD
    # ========================================================

    final_mask = (
        final_probability
        >= MASK_THRESHOLD
    ).astype(
        np.uint8
    )

    # hard mask again
    final_mask[
        ~ocean_mask
    ] = 0

    # ========================================================
    # CLEANING
    # ========================================================

    final_mask = clean_mask(
        final_mask,
        ocean_mask,
        MIN_COMPONENT_AREA
    )

    # final safety check
    final_mask[
        ~ocean_mask
    ] = 0

    print(
        "✓ Strict land/coast filtering complete"
    )

    # ========================================================
    # STATS
    # ========================================================

    oil_pixels = int(
        final_mask.sum()
    )

    spill_detected = (
        oil_pixels > 0
    )

    oil_fraction = (
        oil_pixels
        / ocean_pixels
        if ocean_pixels > 0
        else 0.0
    )

    if spill_detected:

        detection_confidence = float(
            final_probability[
                final_mask == 1
            ].mean()
        )

    else:

        detection_confidence = None

    max_candidate_probability = (
        float(
            final_probability[
                ocean_mask
            ].max()
        )
        if ocean_mask.any()
        else 0.0
    )

    # ========================================================
    # POLYGONS
    # ========================================================

    polygon_objects = []
    geojson_features = []

    for geometry, value in shapes(
        final_mask,
        mask=(
            final_mask.astype(bool)
            & ocean_mask
        ),
        transform=transform
    ):

        if int(value) != 1:
            continue

        polygon = shape(
            geometry
        )

        if polygon.is_empty:
            continue

        polygon_objects.append(
            polygon
        )

        polygon_area = calculate_area_km2(
            polygon
        )

        geojson_features.append(
            {
                "type":
                "Feature",

                "geometry":
                mapping(
                    polygon
                ),

                "properties": {
                    "class":
                    "candidate_oil_spill",

                    "area_km2":
                    polygon_area,

                    "timestamp":
                    scene_metadata.get(
                        "acquisition_timestamp"
                    )
                }
            }
        )

    # ========================================================
    # TOTAL AREA + CENTROID
    # ========================================================

    if polygon_objects:

        merged_geometry = unary_union(
            polygon_objects
        )

        spill_area_km2 = calculate_area_km2(
            merged_geometry
        )

        centroid = (
            merged_geometry.centroid
        )

        spill_location = {
            "lat":
            float(
                centroid.y
            ),

            "lon":
            float(
                centroid.x
            )
        }

    else:

        spill_area_km2 = 0.0
        spill_location = None

    # ========================================================
    # GEOJSON
    # ========================================================

    geojson = {
        "type":
        "FeatureCollection",

        "features":
        geojson_features
    }

    geojson_path = (
        OUTPUT_DIR
        / "oil_spill.geojson"
    )

    geojson_path.write_text(
        json.dumps(
            geojson,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # VISUAL OUTPUTS
    # ========================================================

    vv_png = (
        vv
        * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    probability_png = (
        final_probability
        * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    mask_png = (
        final_mask
        * 255
    ).astype(
        np.uint8
    )

    overlay = np.full(
        (
            height,
            width,
            3
        ),
        128,
        dtype=np.uint8
    )

    vv_rgb = cv2.cvtColor(
        vv_png,
        cv2.COLOR_GRAY2BGR
    )

    # Only ocean shown from VV
    overlay[
        ocean_mask
    ] = vv_rgb[
        ocean_mask
    ]

    # Land shown in dark grey
    land_display = (
        satellite_valid_mask
        & buffered_land_mask
    )

    overlay[
        land_display
    ] = (
        70,
        70,
        70
    )

    # Red only in valid ocean
    overlay[
        final_mask == 1
    ] = (
        0,
        0,
        255
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "sentinel1_vv.png"
        ),
        vv_png
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "probability_map.png"
        ),
        probability_png
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "oil_mask.png"
        ),
        mask_png
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "overlay.png"
        ),
        overlay
    )

    # Save masks for debugging
    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "land_mask.png"
        ),
        (
            land_mask.astype(
                np.uint8
            )
            * 255
        )
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / "ocean_mask.png"
        ),
        (
            ocean_mask.astype(
                np.uint8
            )
            * 255
        )
    )

    # ========================================================
    # RESULT JSON
    # ========================================================

    result = {

        "source":
        scene_metadata.get(
            "source",
            "Copernicus Sentinel-1"
        ),

        "source_product":
        scene_metadata.get(
            "source_product"
        ),

        "platform":
        scene_metadata.get(
            "platform"
        ),

        "polarization":
        scene_metadata.get(
            "polarization"
        ),

        "instrument_mode":
        scene_metadata.get(
            "instrument_mode"
        ),

        "acquisition_timestamp":
        scene_metadata.get(
            "acquisition_timestamp"
        ),

        "spill_detected":
        spill_detected,

        "spill_location":
        spill_location,

        "spill_area_km2":
        spill_area_km2,

        "detection_confidence":
        detection_confidence,

        "max_candidate_probability":
        max_candidate_probability,

        "threshold":
        MASK_THRESHOLD,

        "coast_buffer_pixels":
        COAST_BUFFER_PIXELS,

        "satellite_valid_pixels":
        int(
            satellite_valid_mask.sum()
        ),

        "land_pixels":
        int(
            land_mask.sum()
        ),

        "ocean_pixels":
        ocean_pixels,

        "oil_pixels":
        oil_pixels,

        "oil_fraction_of_ocean_area":
        oil_fraction,

        "number_of_polygons":
        len(
            geojson_features
        ),

        "crs":
        str(
            crs
        ),

        "bounds": [
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top
        ],

        "spill_polygon_geojson":
        str(
            geojson_path
        ),

        "input_geotiff":
        str(
            INPUT_TIF
        ),

        "status":
        (
            "candidate_spill_detected"
            if spill_detected
            else
            "no_reliable_candidate_detected"
        ),

        "note":
        (
            "Land and coastline pixels are excluded. "
            "Remaining detections are ocean-only candidate "
            "oil-spill regions and require validation."
        )
    }

    result_path = (
        OUTPUT_DIR
        / "result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print(
        "MEMBER 1 FINAL OUTPUT"
    )
    print("=" * 72)

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print()
    print(
        "Outputs saved:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 72)


if __name__ == "__main__":

    main()