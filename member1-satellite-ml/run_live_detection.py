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
# ENV + PATHS
# ============================================================

load_dotenv()

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
# DEFAULT AOI
# ============================================================

DEFAULT_BBOX = [
    -90.5,
    27.5,
    -89.5,
    28.5
]


# ============================================================
# AREA INPUT
# ============================================================

def get_user_bbox():

    print()
    print("=" * 70)
    print("SELECT AREA TO ANALYSE")
    print("=" * 70)

    print(
        "Press ENTER / choose n to use default Gulf of Mexico AOI."
    )

    choice = input(
        "Use custom area? (y/n): "
    ).strip().lower()

    if choice != "y":

        print(
            "✓ Using default AOI:",
            DEFAULT_BBOX
        )

        return DEFAULT_BBOX

    try:

        west = float(
            input("West longitude: ").strip()
        )

        south = float(
            input("South latitude: ").strip()
        )

        east = float(
            input("East longitude: ").strip()
        )

        north = float(
            input("North latitude: ").strip()
        )

    except ValueError:

        raise ValueError(
            "Coordinates must be numeric."
        )

    if west >= east:

        raise ValueError(
            "West longitude must be smaller than East longitude."
        )

    if south >= north:

        raise ValueError(
            "South latitude must be smaller than North latitude."
        )

    bbox = [
        west,
        south,
        east,
        north
    ]

    print(
        "✓ Custom AOI:",
        bbox
    )

    return bbox


# ============================================================
# TIME INPUT
# ============================================================

def parse_date(
    value
):

    try:

        return datetime.strptime(
            value,
            "%Y-%m-%d"
        ).replace(
            tzinfo=timezone.utc
        )

    except ValueError:

        raise ValueError(
            "Date format must be YYYY-MM-DD."
        )


def get_time_range():

    print()
    print("=" * 70)
    print("SELECT TIME RANGE")
    print("=" * 70)

    print(
        "1 = Latest available scene"
    )

    print(
        "2 = Exact date range"
    )

    choice = input(
        "Choose option (1/2): "
    ).strip()

    # --------------------------------------------------------
    # OPTION 1 — LATEST
    # --------------------------------------------------------

    if choice in (
        "",
        "1"
    ):

        days_text = input(
            "Search previous how many days? "
            "(default 14): "
        ).strip()

        if not days_text:

            days = 14

        else:

            try:

                days = int(
                    days_text
                )

            except ValueError:

                raise ValueError(
                    "Days must be an integer."
                )

            if days <= 0:

                raise ValueError(
                    "Days must be greater than 0."
                )

        end = datetime.now(
            timezone.utc
        )

        start = (
            end
            - timedelta(days=days)
        )

        return {
            "mode":
            "latest",

            "start":
            start,

            "end":
            end
        }

    # --------------------------------------------------------
    # OPTION 2 — EXACT RANGE
    # --------------------------------------------------------

    elif choice == "2":

        print()
        print(
            "Enter dates in YYYY-MM-DD format."
        )

        start_text = input(
            "Start date: "
        ).strip()

        end_text = input(
            "End date: "
        ).strip()

        start = parse_date(
            start_text
        )

        end = parse_date(
            end_text
        )

        # Include entire end date
        end = (
            end
            + timedelta(
                days=1
            )
            - timedelta(
                seconds=1
            )
        )

        if start >= end:

            raise ValueError(
                "Start date must be before End date."
            )

        return {
            "mode":
            "exact",

            "start":
            start,

            "end":
            end
        }

    else:

        raise ValueError(
            "Choose 1 or 2."
        )


# ============================================================
# SENTINEL-1 SEARCH
# ============================================================

def search_scene(
    bbox,
    time_range
):

    start = time_range[
        "start"
    ]

    end = time_range[
        "end"
    ]

    payload = {

        "collections": [
            "sentinel-1-grd"
        ],

        "bbox":
        bbox,

        "datetime":
        (
            f"{start.isoformat()}/"
            f"{end.isoformat()}"
        ),

        "limit":
        100
    }

    print()
    print("=" * 70)
    print("STEP 1 — SEARCHING SENTINEL-1")
    print("=" * 70)

    print(
        "AOI:",
        bbox
    )

    print(
        "From:",
        start.isoformat()
    )

    print(
        "To:",
        end.isoformat()
    )

    response = requests.post(
        STAC_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

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
            "No Sentinel-1 scenes found for this AOI/date range."
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
            "No Sentinel-1 IW + VV scene found."
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

    selected = usable[0]

    props = selected.get(
        "properties",
        {}
    )

    print()
    print(
        f"✓ Found {len(usable)} usable scene(s)"
    )

    print(
        "✓ Selected most recent scene in requested period"
    )

    print(
        "ID:",
        selected.get("id")
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
        "Mode:",
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

    return selected


# ============================================================
# AUTH TOKEN
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
            "SH_CLIENT_ID / SH_CLIENT_SECRET missing from .env"
        )

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

    return (
        response
        .json()
        ["access_token"]
    )


# ============================================================
# DOWNLOAD SELECTED SCENE AOI
# ============================================================

def download_scene_aoi(
    scene,
    bbox,
    requested_time_range
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
            "Scene timestamp missing."
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

                "bbox":
                bbox,

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
    print("STEP 2 — FETCHING GEOCODED VV AOI")
    print("=" * 70)

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
        bbox,

        "requested_time_mode":
        requested_time_range.get(
            "mode"
        ),

        "requested_start":
        requested_time_range[
            "start"
        ].isoformat(),

        "requested_end":
        requested_time_range[
            "end"
        ].isoformat()
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "✓ GeoTIFF saved:"
    )

    print(
        OUTPUT_TIF
    )


# ============================================================
# RUN MODEL
# ============================================================

def run_inference():

    print()
    print("=" * 70)
    print("STEP 3 — RUNNING U-NET")
    print("=" * 70)

    env = os.environ.copy()

    env[
        "PYTHONPATH"
    ] = str(
        BASE_DIR
    )

    subprocess.run(
        [
            sys.executable,
            str(
                BASE_DIR
                / "src"
                / "satellite"
                / "sentinel1_inference.py"
            )
        ],
        env=env,
        check=True
    )


# ============================================================
# SUMMARY
# ============================================================

def print_final_summary():

    if not RESULT_PATH.exists():

        print(
            "Result file not found."
        )

        return

    result = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    print()
    print("=" * 70)
    print("OILTRACE AI — ANALYSIS SUMMARY")
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
        "Candidate spill:",
        "YES"
        if detected
        else "NO"
    )

    if detected:

        print(
            "Area:",
            f"{result.get('spill_area_km2', 0):.2f} km²"
        )

        confidence = result.get(
            "detection_confidence"
        )

        if confidence is not None:

            print(
                "Confidence:",
                f"{confidence * 100:.1f}%"
            )

        print(
            "Polygons:",
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
                "Centroid:",
                f"{location['lat']:.4f}, "
                f"{location['lon']:.4f}"
            )

    else:

        print(
            "No reliable candidate retained "
            "after post-processing."
        )

    print(
        "Status:",
        result.get(
            "status"
        )
    )

    print("=" * 70)


# ============================================================
# OPEN OVERLAY
# ============================================================

def open_overlay():

    if not OVERLAY_PATH.exists():
        return

    system = platform.system()

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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("OILTRACE AI — SENTINEL-1 AREA ANALYSIS")
    print("=" * 70)

    try:

        bbox = (
            get_user_bbox()
        )

        time_range = (
            get_time_range()
        )

        scene = search_scene(
            bbox,
            time_range
        )

        download_scene_aoi(
            scene,
            bbox,
            time_range
        )

        run_inference()

        print_final_summary()

        open_overlay()

        print()
        print(
            "✓ ANALYSIS COMPLETE"
        )

    except KeyboardInterrupt:

        print(
            "\nAnalysis stopped."
        )

    except Exception as error:

        print()
        print(
            "ANALYSIS FAILED:"
        )

        print(
            error
        )


if __name__ == "__main__":

    main()