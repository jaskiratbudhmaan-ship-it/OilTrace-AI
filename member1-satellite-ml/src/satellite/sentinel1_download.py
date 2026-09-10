from pathlib import Path
import requests
from tqdm import tqdm

from cdse_auth import get_access_token


BASE_DIR = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = BASE_DIR / "data" / "sentinel1"

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def download_product(product_uuid, product_name):

    access_token = get_access_token()

    url = (
        "https://download.dataspace.copernicus.eu/"
        f"odata/v1/Products({product_uuid})/$value"
    )

    output_path = DOWNLOAD_DIR / f"{product_name}.zip"

    existing_size = 0

    if output_path.exists():
        existing_size = output_path.stat().st_size

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(
            f"Resuming from "
            f"{existing_size / (1024*1024):.1f} MB"
        )

    print("\nDownloading Sentinel-1 product...")
    print("Product:", product_name)
    print("UUID:", product_uuid)

    try:

        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 600)
        ) as response:

            if response.status_code not in (200, 206):
                response.raise_for_status()

            total_remaining = int(
                response.headers.get(
                    "content-length",
                    0
                )
            )

            total_size = existing_size + total_remaining

            mode = "ab" if existing_size > 0 else "wb"

            with open(
                output_path,
                mode
            ) as file:

                with tqdm(
                    total=total_size,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    desc="Download"
                ) as progress:

                    for chunk in response.iter_content(
                        chunk_size=4 * 1024 * 1024
                    ):

                        if not chunk:
                            continue

                        file.write(chunk)

                        progress.update(
                            len(chunk)
                        )

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ReadTimeout
    ) as error:

        print("\n\nConnection interrupted.")
        print(error)

        print("\nPartial download preserved.")
        print(
            "Run the SAME command again "
            "to continue downloading."
        )

        return None

    print("\n✓ Download complete")

    print("Saved:")
    print(output_path)

    print(
        "Final size:",
        f"{output_path.stat().st_size / (1024**2):.1f} MB"
    )

    return output_path


if __name__ == "__main__":

    product_uuid = (
        "9e9d2f0c-7c3c-4b8c-a907-f7ab30acfb50"
    )

    product_name = (
        "S1D_IW_GRDH_1SDV_"
        "20260908T000923_"
        "20260908T000948_"
        "004479_0084FE_"
        "D121_COG.SAFE"
    )

    download_product(
        product_uuid,
        product_name
    )