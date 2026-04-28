import os
import json
import time
import requests
import numpy as np
from pathlib import Path

# ------------------------------------
# CONFIG
# ------------------------------------
RENTCAST_API_KEY = os.environ.get("RENTCAST_API_KEY", "")
BASE_URL = "https://api.rentcast.io/v1"
CACHE_PATH = Path("outputs/rentcast_cache.json")
REQUEST_DELAY = 0.5  # seconds between API calls to avoid rate limiting


# ------------------------------------
# CACHE HELPERS
# ------------------------------------
def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _cache_key(endpoint, params):
    """Stable string key for a given request."""
    sorted_params = sorted((k, str(v)) for k, v in params.items() if v is not None)
    return f"{endpoint}|{'|'.join(f'{k}={v}' for k, v in sorted_params)}"


# ------------------------------------
# CORE REQUEST
# ------------------------------------
def _get(endpoint, params):
    """
    Make a cached GET request to RentCast API.
    Returns parsed JSON dict or None on failure.
    """
    cache = _load_cache()
    key = _cache_key(endpoint, params)

    if key in cache:
        return cache[key]

    headers = {"X-Api-Key": RENTCAST_API_KEY, "Accept": "application/json"}
    url = f"{BASE_URL}/{endpoint}"

    try:
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            cache[key] = data
            _save_cache(cache)
            return data

        elif response.status_code == 429:
            print("RentCast rate limit hit. Waiting 10 seconds...")
            time.sleep(10)
            return None

        elif response.status_code == 401:
            print("RentCast API key invalid or missing.")
            return None

        else:
            print(f"RentCast error {response.status_code}: {response.text[:200]}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"RentCast request failed: {e}")
        return None


# ------------------------------------
# AVM: SALE VALUE ESTIMATE
# ------------------------------------
def get_avm_estimate(address, city, state, zip_code=None,
                     bedrooms=None, bathrooms=None, sqft=None):
    """
    Get RentCast's automated valuation (AVM) for a property.

    Returns dict with keys:
        price         - point estimate (int)
        price_low     - lower bound
        price_high    - upper bound
        confidence    - 'High' / 'Medium' / 'Low'
    Or None if the API call fails or no estimate is available.
    """
    params = {
        "address":       address,
        "city":          city,
        "state":         state,
        "zipCode":       zip_code,
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms,
        "squareFootage": sqft,
        "propertyType":  "Single Family",  # default; overridden if property lookup succeeds
    }
    params = {k: v for k, v in params.items() if v is not None}

    data = _get("avm/value", params)

    if not data or "price" not in data:
        return None

    return {
        "price":      data.get("price"),
        "price_low":  data.get("priceRangeLow"),
        "price_high": data.get("priceRangeHigh"),
        "confidence": data.get("confidence", "Unknown"),
    }


# ------------------------------------
# PROPERTY DETAILS LOOKUP
# ------------------------------------
def get_property_details(address, city, state, zip_code=None):
    """
    Look up property characteristics from RentCast.

    Returns dict with keys (all may be None if not available):
        year_built, property_type, bedrooms, bathrooms,
        sqft, lot_size, last_sale_price, last_sale_date
    Or None if the property is not found.
    """
    params = {
        "address": address,
        "city":    city,
        "state":   state,
        "zipCode": zip_code,
    }
    params = {k: v for k, v in params.items() if v is not None}

    data = _get("properties", params)

    if not data:
        return None

    # API returns a list; take the first match
    if isinstance(data, list):
        if len(data) == 0:
            return None
        data = data[0]

    return {
        "year_built":       data.get("yearBuilt"),
        "property_type":    data.get("propertyType"),
        "bedrooms":         data.get("bedrooms"),
        "bathrooms":        data.get("bathrooms"),
        "sqft":             data.get("squareFootage"),
        "lot_size":         data.get("lotSize"),
        "last_sale_price":  data.get("lastSalePrice"),
        "last_sale_date":   data.get("lastSaleDate"),
    }


# ------------------------------------
# MARKET STATS BY ZIP
# ------------------------------------
def get_market_stats(zip_code, state=None):
    """
    Get market-level statistics for a zip code.

    Returns dict with keys:
        median_sale_price, median_days_on_market,
        sale_count, price_appreciation_1yr
    Or None if not available.
    """
    params = {"zipCode": zip_code}
    if state:
        params["state"] = state

    data = _get("markets", params)

    if not data:
        return None

    sale = data.get("saleData", {})

    return {
        "median_sale_price":      sale.get("averagePrice"),
        "median_days_on_market":  sale.get("averageDaysOnMarket"),
        "sale_count":             sale.get("totalListings"),
        "price_per_sqft_market":  sale.get("averagePricePerSquareFoot"),
    }


# ------------------------------------
# COMBINED ENRICHMENT (single call)
# ------------------------------------
def enrich_property(address, city, state, zip_code=None,
                    bedrooms=None, bathrooms=None, sqft=None):
    """
    Full enrichment for a single property at inference time.
    Tries property details + AVM in one go.

    Returns:
        details  - property characteristics (year_built, type, etc.)
        avm      - price estimate dict
    """
    details = get_property_details(address, city, state, zip_code)
    avm = get_avm_estimate(
        address, city, state, zip_code,
        bedrooms=bedrooms, bathrooms=bathrooms, sqft=sqft
    )
    return details, avm
