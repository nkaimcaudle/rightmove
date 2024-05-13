import requests
import json
import argparse

import numpy as np


def fetch_property(property_id):
    uri = f"https://www.rightmove.co.uk/properties/{property_id}"
    proxies = [
        {},
        {
            "http": "http://192.168.0.64:8008",
        },
        {
            "http": "http://192.168.0.64:8008",
        },
        {
            "http": "http://192.168.0.64:8009",
        },
    ]
    proxy = np.random.choice(proxies)
    response = requests.get(uri, proxies=proxy)
    return response


def get_json(property_id) -> dict:
    response = fetch_property(property_id)
    if response.status_code == 200:
        line = [x for x in response.text.split("\n") if "PAGE_MODEL" in x][0]
        model = line[line.find("{"): line.rfind("}") + 1]
        dct = json.loads(model)
        return dct
    return None


def run_property(property_id) -> dict:
    dct = get_json(property_id)
    if dct:
        with open(f"data/property.{property_id}.json", "w") as f:
            json.dump(dct, f)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("property_id", type=str)
    args = parser.parse_args()
    print(args)

    run_property(args.property_id)

    exit(0)
