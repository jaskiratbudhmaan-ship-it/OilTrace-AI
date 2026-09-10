import os
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = BASE_DIR / "data" / "sentinel1" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/process/v1"
)


def get_token():

    client_id = os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SH_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "SH_CLIENT_ID or SH_CLIENT_SECRET missing in .env"
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["access_token"]


def download_vv_aoi():

    token = get_token()

    # Gulf of Mexico AOI
    bbox = [
        -90.5,
        27.5,
        -89.5,
        28.5
    ]

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
                "bbox": bbox,
                "properties": {
                    "crs":
                    "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },

            "data": [
                {
                    "type": "sentinel-1-grd",

                    "dataFilter": {

                        "timeRange": {
                            "from":
                            "2026-09-08T00:09:00Z",

                            "to":
                            "2026-09-08T00:10:30Z"
                        },

                        "acquisitionMode": "IW",

                        "polarization":
                        "DV"
                    },

                    "processing": {

                        "orthorectify": True,

                        "backCoeff":
                        "SIGMA0_ELLIPSOID"
                    }
                }
            ]
        },

        "output": {

            "width": 1024,

            "height": 1024,

            "responses": [
                {
                    "identifier": "default",

                    "format": {
                        "type":
                        "image/tiff"
                    }
                }
            ]
        },

        "evalscript": evalscript
    }

    print(
        "Requesting georeferenced "
        "Sentinel-1 VV AOI..."
    )

    response = requests.post(

        PROCESS_URL,

        headers={
            "Authorization":
            f"Bearer {token}"
        },

        json=request_body,

        timeout=180
    )

    response.raise_for_status()

    output_path = (
        OUTPUT_DIR /
        "sentinel1_gulf_vv_geocoded.tif"
    )

    output_path.write_bytes(
        response.content
    )

    print("✓ Sentinel-1 AOI downloaded")
    print("Saved:", output_path)


if __name__ == "__main__":

    download_vv_aoi()