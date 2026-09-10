from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod
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
# SENTINEL-1 SCENE METADATA
# ============================================================

SOURCE_PRODUCT = (
    "S1D_IW_GRDH_1SDV_"
    "20260908T000923_"
    "20260908T000948_"
    "004479_0084FE_D121_COG.SAFE"
)

ACQUISITION_TIMESTAMP = (
    "2026-09-08T00:09:23Z"
)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 256

MIN_COMPONENT_AREA = 50

GEOD = Geod(
    ellps="WGS84"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sar(
    image,
    valid_mask
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    valid_mask = (
        valid_mask.astype(bool)
        & np.isfinite(image)
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    if not valid_mask.any():
        return output

    values = image[
        valid_mask
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
        valid_mask
    ] = (
        clipped[valid_mask]
        - low
    ) / (
        high - low
    )

    output[
        ~valid_mask
    ] = 0

    return output


# ============================================================
# MODEL
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
            .sigmoid(logits)
            [0, 0]
            .cpu()
            .numpy()
        )

    return probability


# ============================================================
# MASK CLEANING
# ============================================================

def clean_mask(
    mask,
    valid_mask,
    min_area=50
):

    mask = mask.astype(
        np.uint8
    )

    valid_mask = valid_mask.astype(
        bool
    )

    mask[
        ~valid_mask
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

    invalid_mask = (
        ~valid_mask
    ).astype(
        np.uint8
    )

    dilated_invalid = cv2.dilate(

        invalid_mask,

        np.ones(
            (5, 5),
            np.uint8
        ),

        iterations=1

    ).astype(
        bool
    )

    height, width = mask.shape

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

        touches_image_border = (

            component[
                0,
                :
            ].any()

            or

            component[
                height - 1,
                :
            ].any()

            or

            component[
                :,
                0
            ].any()

            or

            component[
                :,
                width - 1
            ].any()
        )

        touches_invalid_boundary = (
            component
            & dilated_invalid
        ).any()

        if (
            touches_image_border
            or touches_invalid_boundary
        ):

            continue

        cleaned[
            component
        ] = 1

    cleaned[
        ~valid_mask
    ] = 0

    return cleaned


# ============================================================
# GEODESIC AREA
# ============================================================

def calculate_area_km2(
    geometry
):

    area_m2, _ = (
        GEOD.geometry_area_perimeter(
            geometry
        )
    )

    area_m2 = abs(
        area_m2
    )

    return (
        area_m2
        / 1_000_000
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "OILTRACE AI - MEMBER 1 REAL SENTINEL-1 PIPELINE"
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_TIF.exists():

        raise FileNotFoundError(
            INPUT_TIF
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    print(
        "✓ U-Net model loaded"
    )


    # --------------------------------------------------------
    # READ SENTINEL-1 GEOTIFF
    # --------------------------------------------------------

    with rasterio.open(
        INPUT_TIF
    ) as src:

        # Band 1 = VV
        vv = src.read(
            1
        )

        # Band 2 = dataMask
        data_mask = src.read(
            2
        )

        transform = (
            src.transform
        )

        crs = (
            src.crs
        )

        width = (
            src.width
        )

        height = (
            src.height
        )

        bounds = (
            src.bounds
        )


    print(
        "✓ Sentinel-1 VV GeoTIFF loaded"
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


    # --------------------------------------------------------
    # VALID SAR AREA
    # --------------------------------------------------------

    valid_mask = (
        data_mask > 0
    )

    valid_pixels = int(
        valid_mask.sum()
    )

    print(
        "Valid pixels:",
        valid_pixels
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    vv = normalize_sar(
        vv,
        valid_mask
    )

    print(
        "✓ SAR normalization complete"
    )


    # --------------------------------------------------------
    # TILE-WISE MODEL INFERENCE
    # --------------------------------------------------------

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

            tile_mask = valid_mask[
                y:y + valid_h,
                x:x + valid_w
            ]


            # no satellite data
            if not tile_mask.any():

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


            probability[
                ~tile_mask
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
        "Tiles processed by model:",
        model_tiles
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    final_mask = (
        final_probability
        >= MASK_THRESHOLD
    ).astype(
        np.uint8
    )

    final_mask[
        ~valid_mask
    ] = 0


    # --------------------------------------------------------
    # POST PROCESSING
    # --------------------------------------------------------

    final_mask = clean_mask(
        final_mask,
        valid_mask,
        MIN_COMPONENT_AREA
    )

    print(
        "✓ False-positive cleaning complete"
    )


    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    oil_pixels = int(
        final_mask.sum()
    )

    spill_detected = (
        oil_pixels > 0
    )


    oil_fraction = (

        oil_pixels
        / valid_pixels

        if valid_pixels > 0

        else 0.0
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

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
                valid_mask
            ].max()
        )

        if valid_mask.any()

        else 0.0
    )


    # --------------------------------------------------------
    # CREATE GEO POLYGONS
    # --------------------------------------------------------

    polygon_objects = []

    geojson_features = []


    for geometry, value in shapes(

        final_mask,

        mask=(
            final_mask.astype(bool)
            & valid_mask
        ),

        transform=transform

    ):

        if int(
            value
        ) != 1:

            continue


        polygon = shape(
            geometry
        )


        if polygon.is_empty:

            continue


        polygon_objects.append(
            polygon
        )


        polygon_area = (
            calculate_area_km2(
                polygon
            )
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
                    ACQUISITION_TIMESTAMP
                }
            }
        )


    # --------------------------------------------------------
    # TOTAL AREA + CENTROID
    # --------------------------------------------------------

    if polygon_objects:

        merged_geometry = unary_union(
            polygon_objects
        )

        spill_area_km2 = (
            calculate_area_km2(
                merged_geometry
            )
        )

        centroid = (
            merged_geometry.centroid
        )

        centroid_location = {

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

        centroid_location = None


    # --------------------------------------------------------
    # GEOJSON
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VISUAL OUTPUTS
    # --------------------------------------------------------

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


    overlay[
        valid_mask
    ] = vv_rgb[
        valid_mask
    ]


    # red = candidate oil
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


    # --------------------------------------------------------
    # FINAL MEMBER 1 OUTPUT
    # --------------------------------------------------------

    result = {

        "source":
        "Copernicus Sentinel-1",

        "source_product":
        SOURCE_PRODUCT,

        "polarization":
        "VV",

        "acquisition_timestamp":
        ACQUISITION_TIMESTAMP,

        "spill_detected":
        spill_detected,

        "spill_location":
        centroid_location,

        "spill_area_km2":
        spill_area_km2,

        "detection_confidence":
        detection_confidence,

        "max_candidate_probability":
        max_candidate_probability,

        "threshold":
        MASK_THRESHOLD,

        "valid_pixels":
        valid_pixels,

        "oil_pixels":
        oil_pixels,

        "oil_fraction_of_valid_area":
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
            else "no_reliable_candidate_detected"
        ),

        "note":
        (
            "Satellite model output only. "
            "Detected regions are candidate oil-spill "
            "regions and require further validation."
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


    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "MEMBER 1 FINAL OUTPUT"
    )

    print(
        "=" * 72
    )


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

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()