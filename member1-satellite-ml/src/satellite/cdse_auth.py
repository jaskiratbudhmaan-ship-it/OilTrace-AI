import os
import requests
from dotenv import load_dotenv


load_dotenv()


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)


def get_access_token():

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise ValueError(
            "CDSE_USERNAME or CDSE_PASSWORD missing in .env"
        )

    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password
    }

    response = requests.post(
        TOKEN_URL,
        data=data,
        timeout=60
    )

    response.raise_for_status()

    token_data = response.json()

    return token_data["access_token"]


if __name__ == "__main__":

    token = get_access_token()

    print("✓ Copernicus authentication successful")
    print("Access token received successfully.")