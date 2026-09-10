import requests


ODATA_URL = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
)


def find_product_by_name(product_name):

    # STAC ID normally does not contain .SAFE
    if not product_name.upper().endswith(".SAFE"):
        product_name = product_name + ".SAFE"

    params = {
        "$filter": f"Name eq '{product_name}'",
        "$select": "Id,Name,ContentDate",
    }

    print("\nSearching OData catalogue...")
    print("Product:", product_name)

    response = requests.get(
        ODATA_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    products = data.get("value", [])

    if not products:
        return None

    return products[0]


if __name__ == "__main__":

    product_name = input(
        "Paste Sentinel-1 product ID/name: "
    ).strip()

    product = find_product_by_name(product_name)

    if product:

        print("\n✓ PRODUCT FOUND")

        print(
            "UUID:",
            product["Id"]
        )

        print(
            "Name:",
            product["Name"]
        )

        print(
            "Content Date:",
            product.get("ContentDate")
        )

    else:

        print("\n✗ Product not found.")