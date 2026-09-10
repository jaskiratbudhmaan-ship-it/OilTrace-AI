import requests
from datetime import datetime, timedelta, timezone


STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"


def search_sentinel1(
    bbox,
    days_back=7,
    limit=10
):
    """
    Search recent Sentinel-1 GRD satellite products.

    bbox format:
    [west_longitude, south_latitude,
     east_longitude, north_latitude]
    """

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    payload = {
        "collections": ["sentinel-1-grd"],

        "bbox": bbox,

        "datetime": (
            f"{start_time.isoformat()}/"
            f"{end_time.isoformat()}"
        ),

        "limit": limit
    }

    print("\nSearching Copernicus Sentinel-1...")
    print("Bounding Box:", bbox)
    print("Time Range:", start_time, "to", end_time)

    response = requests.post(
        STAC_SEARCH_URL,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    print("\nProducts found:", len(features))

    return features


def display_products(products):

    if not products:
        print("\nNo Sentinel-1 products found.")
        return

    print("\n" + "=" * 70)
    print("RECENT SENTINEL-1 PRODUCTS")
    print("=" * 70)

    for index, product in enumerate(products, start=1):

        properties = product.get("properties", {})

        print("\nProduct", index)

        print(
            "ID:",
            product.get("id")
        )

        print(
            "Date:",
            properties.get("datetime")
        )

        print(
            "Platform:",
            properties.get("platform")
        )

        print(
            "Polarization:",
            properties.get(
                "sar:polarizations"
            )
        )

        print(
            "Instrument mode:",
            properties.get(
                "sar:instrument_mode"
            )
        )

        print(
            "Bounding Box:",
            product.get("bbox")
        )
        print(
           "Assets:",
           list(product.get("assets", {}).keys())
                                                  )
        assets = product.get("assets", {})

        if "vv" in assets:
           print(
        "VV HREF:",
        assets["vv"].get("href")
    )


if __name__ == "__main__":

    # Example maritime region:
    # Gulf of Mexico test AOI

    bbox = [
        -90.5,   # west longitude
        27.5,    # south latitude
        -89.5,   # east longitude
        28.5     # north latitude
    ]

    products = search_sentinel1(
        bbox=bbox,
        days_back=14,
        limit=5
    )

    display_products(products)