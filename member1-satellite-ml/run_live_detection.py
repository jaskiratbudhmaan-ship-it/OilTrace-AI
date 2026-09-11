from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import os
import platform
import subprocess
import sys

import requests
from dotenv import load_dotenv


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "sentinel1"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_TIF = (
    PROCESSED_DIR
    / "sentinel1_gulf_vv_geocoded.tif"
)

METADATA_FILE = (
    PROCESSED_DIR
    / "latest_scene.json"
)

RESULT_PATH = (
    BASE_DIR
    / "outputs"
    / "sentinel1_real"
    / "result.json"
)

OVERLAY_PATH = (
    BASE_DIR
    / "outputs"
    / "sentinel1_real"
    / "overlay.png"
)


# ============================================================
# ANALYSIS AREA
# Gulf of Mexico AOI
# ============================================================

BBOX = [
    -90.5,   # West
    27.5,    # South
    -89.5,   # East
    28.5     # North
]


# ============================================================
# COPERNICUS ENDPOINTS
# ============================================================

STAC_URL = (
    "https://stac.dataspace.copernicus.eu/v1/search"
)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/process/v1"
)


# ============================================================
# STEP 1
# SEARCH LATEST SENTINEL-1 SCENE
# ============================================================

def search_latest_scene():

    now = datetime.now(
        timezone.utc
    )

    start = (
        now
        - timedelta(days=14)
    )

    payload = {

        "collections": [
            "sentinel-1-grd"
        ],

        "bbox": BBOX,

        "datetime": (
            f"{start.isoformat()}/"
            f"{now.isoformat()}"
        ),

        "limit": 20
    }

    print()
    print("=" * 70)
    print("STEP 1 — SEARCHING LATEST SENTINEL-1")
    print("=" * 70)

    try:

        response = requests.post(
            STAC_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            f"Sentinel-1 STAC search failed: {error}"
        )

    products = (
        response
        .json()
        .get(
            "features",
            []
        )
    )

    if not products:

        raise RuntimeError(
            "No recent Sentinel-1 products found "
            "for the selected AOI."
        )

    usable = []

    for item in products:

        props = item.get(
            "properties",
            {}
        )

        mode = props.get(
            "sar:instrument_mode"
        )

        polarizations = props.get(
            "sar:polarizations",
            []
        )

        if (
            mode == "IW"
            and "VV" in polarizations
        ):

            usable.append(
                item
            )

    if not usable:

        raise RuntimeError(
            "No recent Sentinel-1 IW + VV scene found."
        )

    usable.sort(
        key=lambda item:
        item.get(
            "properties",
            {}
        ).get(
            "datetime",
            ""
        ),
        reverse=True
    )

    latest = usable[0]

    props = latest.get(
        "properties",
        {}
    )

    print(
        "✓ Latest scene found"
    )

    print(
        "ID:",
        latest.get("id")
    )

    print(
        "Date:",
        props.get("datetime")
    )

    print(
        "Platform:",
        props.get("platform")
    )

    print(
        "Instrument mode:",
        props.get(
            "sar:instrument_mode"
        )
    )

    print(
        "Polarization:",
        props.get(
            "sar:polarizations"
        )
    )

    return latest


# ============================================================
# STEP 2
# SENTINEL HUB ACCESS TOKEN
# ============================================================

def get_process_token():

    client_id = os.getenv(
        "SH_CLIENT_ID"
    )

    client_secret = os.getenv(
        "SH_CLIENT_SECRET"
    )

    if (
        not client_id
        or not client_secret
    ):

        raise RuntimeError(
            "SH_CLIENT_ID or SH_CLIENT_SECRET "
            "missing in .env"
        )

    try:

        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type":
                "client_credentials",

                "client_id":
                client_id,

                "client_secret":
                client_secret,
            },
            timeout=60
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            f"Copernicus authentication failed: {error}"
        )

    return (
        response
        .json()
        ["access_token"]
    )


# ============================================================
# STEP 3
# DOWNLOAD LATEST GEOCODED VV AOI
# ============================================================

def download_scene_aoi(
    scene
):

    props = scene.get(
        "properties",
        {}
    )

    scene_datetime = props.get(
        "datetime"
    )

    if not scene_datetime:

        raise RuntimeError(
            "Scene acquisition timestamp missing."
        )

    acquisition = (
        datetime
        .fromisoformat(
            scene_datetime.replace(
                "Z",
                "+00:00"
            )
        )
    )

    # Small time window around selected acquisition
    start_time = (
        acquisition
        - timedelta(minutes=1)
    )

    end_time = (
        acquisition
        + timedelta(minutes=1)
    )

    token = (
        get_process_token()
    )

    evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: ["VV", "dataMask"],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [
            sample.VV,
            sample.dataMask
        ];
    }
    """

    request_body = {

        "input": {

            "bounds": {

                "bbox": BBOX,

                "properties": {
                    "crs":
                    "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },

            "data": [

                {

                    "type":
                    "sentinel-1-grd",

                    "dataFilter": {

                        "timeRange": {

                            "from":
                            start_time
                            .isoformat()
                            .replace(
                                "+00:00",
                                "Z"
                            ),

                            "to":
                            end_time
                            .isoformat()
                            .replace(
                                "+00:00",
                                "Z"
                            )
                        },

                        "acquisitionMode":
                        "IW",

                        "polarization":
                        "DV"
                    },

                    "processing": {

                        "orthorectify":
                        True,

                        "backCoeff":
                        "SIGMA0_ELLIPSOID"
                    }
                }
            ]
        },

        "output": {

            "width":
            1024,

            "height":
            1024,

            "responses": [

                {

                    "identifier":
                    "default",

                    "format": {
                        "type":
                        "image/tiff"
                    }
                }
            ]
        },

        "evalscript":
        evalscript
    }

    print()
    print("=" * 70)
    print("STEP 2 — DOWNLOADING GEOCODED VV AOI")
    print("=" * 70)

    try:

        response = requests.post(
            PROCESS_URL,
            headers={
                "Authorization":
                f"Bearer {token}"
            },
            json=request_body,
            timeout=240
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise RuntimeError(
            f"Sentinel Hub Process API failed: {error}"
        )

    OUTPUT_TIF.write_bytes(
        response.content
    )

    metadata = {

        "source":
        "Copernicus Sentinel-1",

        "source_product":
        scene.get("id"),

        "acquisition_timestamp":
        scene_datetime,

        "platform":
        props.get(
            "platform"
        ),

        "polarization":
        props.get(
            "sar:polarizations"
        ),

        "instrument_mode":
        props.get(
            "sar:instrument_mode"
        ),

        "scene_bbox":
        scene.get(
            "bbox"
        ),

        "analysis_bbox":
        BBOX
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "✓ Latest georeferenced VV AOI saved"
    )

    print(
        OUTPUT_TIF
    )


# ============================================================
# STEP 4
# RUN TRAINED U-NET INFERENCE
# ============================================================

def run_inference():

    print()
    print("=" * 70)
    print("STEP 3 — RUNNING U-NET SPILL DETECTION")
    print("=" * 70)

    env = os.environ.copy()

    env[
        "PYTHONPATH"
    ] = str(
        BASE_DIR
    )

    inference_file = (
        BASE_DIR
        / "src"
        / "satellite"
        / "sentinel1_inference.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(
                inference_file
            )
        ],
        env=env,
        check=True
    )


# ============================================================
# FINAL CLEAN SUMMARY
# ============================================================

def print_final_summary():

    if not RESULT_PATH.exists():

        print()
        print(
            "Result file not found:"
        )

        print(
            RESULT_PATH
        )

        return

    result = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    print()
    print("=" * 70)
    print("OILTRACE AI — LIVE DETECTION SUMMARY")
    print("=" * 70)

    print(
        "Source:",
        result.get(
            "source"
        )
    )

    print(
        "Scene:",
        result.get(
            "source_product"
        )
    )

    print(
        "Platform:",
        result.get(
            "platform"
        )
    )

    print(
        "Acquisition:",
        result.get(
            "acquisition_timestamp"
        )
    )

    detected = result.get(
        "spill_detected",
        False
    )

    print(
        "Candidate spill detected:",
        "YES"
        if detected
        else "NO"
    )

    if detected:

        area = result.get(
            "spill_area_km2",
            0.0
        )

        print(
            "Candidate area:",
            f"{area:.2f} km²"
        )

        confidence = result.get(
            "detection_confidence"
        )

        if confidence is not None:

            print(
                "Detection confidence:",
                f"{confidence * 100:.1f}%"
            )

        print(
            "Detected polygons:",
            result.get(
                "number_of_polygons",
                0
            )
        )

        location = result.get(
            "spill_location"
        )

        if location:

            print(
                "Candidate location:",
                f"{location['lat']:.4f}, "
                f"{location['lon']:.4f}"
            )

    else:

        print(
            "Result:",
            "No reliable candidate oil spill "
            "retained after post-processing."
        )

    print(
        "Status:",
        result.get(
            "status"
        )
    )

    print(
        "Threshold:",
        result.get(
            "threshold"
        )
    )

    print()
    print(
        "Result JSON:"
    )

    print(
        RESULT_PATH
    )

    print()
    print(
        "Overlay:"
    )

    print(
        OVERLAY_PATH
    )

    print("=" * 70)

    print(
        "NOTE: Candidate detections are model outputs "
        "and are not confirmed oil spills."
    )

    print("=" * 70)


# ============================================================
# AUTO OPEN OVERLAY
# ============================================================

def open_overlay():

    if not OVERLAY_PATH.exists():

        print(
            "Overlay file not found:"
        )

        print(
            OVERLAY_PATH
        )

        return

    system = platform.system()

    try:

        if system == "Darwin":

            subprocess.run(
                [
                    "open",
                    str(
                        OVERLAY_PATH
                    )
                ],
                check=False
            )

        elif system == "Windows":

            os.startfile(
                str(
                    OVERLAY_PATH
                )
            )

        elif system == "Linux":

            subprocess.run(
                [
                    "xdg-open",
                    str(
                        OVERLAY_PATH
                    )
                ],
                check=False
            )

    except Exception as error:

        print(
            "Could not automatically open overlay:"
        )

        print(
            error
        )


# ============================================================
# MAIN LIVE PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("OILTRACE AI — LIVE SENTINEL-1 ANALYSIS")
    print("=" * 70)

    print(
        "AOI:",
        BBOX
    )

    try:

        # Latest available Sentinel-1 scene
        scene = (
            search_latest_scene()
        )

        # Get georeferenced VV data
        download_scene_aoi(
            scene
        )

        # Run trained segmentation model
        run_inference()

        # Print human-readable result
        print_final_summary()

        # Automatically open overlay image
        open_overlay()

        print()
        print("=" * 70)
        print("✓ LIVE ANALYSIS COMPLETE")
        print("=" * 70)

    except KeyboardInterrupt:

        print()
        print(
            "Analysis stopped by user."
        )

    except subprocess.CalledProcessError as error:

        print()
        print(
            "Inference process failed."
        )

        print(
            error
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("LIVE ANALYSIS FAILED")
        print("=" * 70)

        print(
            error
        )


if __name__ == "__main__":

    main()