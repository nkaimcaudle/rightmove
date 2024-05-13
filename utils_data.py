import json
from multiprocessing import Pool
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import requests


def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df = df.drop_duplicates(["propertyCard-moreInfoItem href"]).reset_index(drop=True)
    df["filename_postcode"] = p.name.split(".")[1]
    return df


def get_base_data() -> pd.DataFrame:
    with Pool(processes=4) as pool:
        res = pool.map(load_csv, Path("data").glob("*.csv"))
    df = pd.concat(res, ignore_index=True)
    df = df.loc[df["propertyCard-priceValue"].str.contains("£")]
    df["px"] = (
        df["propertyCard-priceValue"]
        .str.replace(",", "")
        .str.replace("£", "")
        .astype(float)
    )
    df = df.drop([x for x in df.columns if "img" in x.lower()], axis=1)
    df = df.drop([x for x in df.columns if "image" in x.lower()], axis=1)
    df = df.drop([x for x in df.columns if "camera" in x.lower()], axis=1)

    df["rightmove_id"] = (
        df["propertyCard-moreInfoItem href"]
        .str.split("/", expand=True)[4]
        .str.replace("#", "")
    )

    df["beds"] = df["text 2"]
    df["bathrooms"] = df["text 3"]

    to_drop = [
        "propertyCard-branchLogo-link href",
        "propertyCard-priceLink href",
        "text 2",
        "text 3",
        "propertyCard-priceValue",
        "no-svg-bed-icon",
        "no-svg-bed-icon href",
        "no-svg-bathroom-icon",
        "no-svg-bathroom-icon href",
        "seperator",
        "seperator 2",
        "propertyCard-branchSummary",
        "propertyCard-headerLink href",
        "propertyCard-contactsPhoneNumber",
        "propertyCard-contactsPhoneNumber href",
        "no-svg-close href",
        "no-svg-floorplan href",
        "propertyCard-moreInfoItem href 3",
        "no-svg-virtualtour href",
        "propertyCard-moreInfoItem href 2",
        "ksc_lozenge",
        "no-svg-chevron-line href",
        "swipe-wrap href",
        "aria-announcer",
        "propertyCard-headerLabel",
        "action href 2",
        "propertyCard-moreInfoNumber",
    ]
    df = df.drop(to_drop, axis=1)

    to_drop = [
        "Land for sale",
        "Block of Apartments",
        "Hotel Room",
        "Park Home",
        "Parking",
        "Mobile Home",
    ]
    df = df.loc[~df["text"].isin(to_drop)]

    df = df.loc[~df["beds"].isna()]
    df = df.loc[~df["bathrooms"].isna()]

    return df.reset_index(drop=True)


def _get_property_data() -> pd.DataFrame:
    res = []
    for p in Path("data").glob("property*json"):
        with open(p, "r") as f:
            dct = json.load(f)
        res.append(dct)
    df = pd.json_normalize(res)
    return df


def get_property_data() -> pd.DataFrame:
    df = _get_property_data()
    to_drop = [
        "auction",
        "customer",
        "branch",
        "email",
        "cookie",
        "misinfo",
        "adinfo",
        "google",
        "featureswitches",
    ]
    for word in to_drop:
        cols = [col for col in df.columns if word in col.lower()]
        df = df.drop(cols, axis=1)

    df["postcode_head"] = df["analyticsInfo.analyticsProperty.postcode"].str.split(
        " ", expand=True
    )[0]

    to_drop = [
        "Land for sale",
        "Block of Apartments",
        "Hotel Room",
        "Park Home",
        "Parking",
        "Mobile Home",
    ]
    df = df.loc[~df["analyticsInfo.analyticsProperty.propertySubType"].isin(to_drop)]

    lon = "analyticsInfo.analyticsProperty.longitude"
    lat = "analyticsInfo.analyticsProperty.latitude"
    mask = (df[lon].values > 25) & (df[lat] < 20)
    df[lon], df[lat] = np.where(mask, df[lat], df[lon]), np.where(
        mask, df[lon], df[lat]
    )

    mask = df[lon].values > -2
    df = df.loc[mask]
    mask = df[lon].values < 0
    df = df.loc[mask]

    mask = df["analyticsInfo.analyticsProperty.maxSizeFt"].lt(200.0).values
    df = df.loc[~mask]
    mask = df["analyticsInfo.analyticsProperty.maxSizeFt"].gt(15e3).values
    df = df.loc[~mask]

    return df.set_index("propertyData.id")


def get_texts_from_floorplan(fname: str) -> list:
    uri = "http://192.168.0.64:8081/text"

    files = {
        "image": open(fname, "rb"),
    }

    response = requests.post(uri, files=files)
    if response.status_code == 200:
        return str(fname), json.loads(response.text)
    else:
        return str(fname), []


def get_texts_from_floorplans():
    res = []
    for p in Path("data").glob("rightmove.floorplan.*"):
        fname, texts = get_texts_from_floorplan(p)
        res.append((fname, texts))
    return res


def get_dict_of_np_hash(dct):
    h = hashlib.sha256()
    for key, value in dct.items():
        h.update(key.encode())
        if isinstance(value, np.ndarray):
            h.update(value.tobytes())
        elif isinstance(value, (int, float)):
            h.update(np.asarray([value]).tobytes())
        else:
            h.update(bytearray(value))
    return h.hexdigest()
