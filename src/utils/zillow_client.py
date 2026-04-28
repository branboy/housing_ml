"""
Zillow property lookup — Zestimate + school rating at inference time.

Uses the same RapidAPI key already configured for the scraper.

Search strategy (tried in order until a Zestimate is found):
  1. Text-based address search  →  direct zpid lookup
  2. Zip search (sold)          →  fuzzy address match  →  detail
  3. Zip search (for_sale)      →  fuzzy address match  →  detail
  4. Zip search (no home_type)  →  fuzzy address match  →  detail (page 1 + 2)

Returns None (silently) if nothing matches — the blend continues unchanged.
Never raises.
"""

import os
import re
import time
import logging
import requests

log = logging.getLogger(__name__)

HOST        = "zillow-scraper-api.p.rapidapi.com"
SEARCH_URL  = f"https://{HOST}/zillow/search/by-zipcode"
TEXT_URL    = f"https://{HOST}/zillow/search"        # text-query endpoint
DETAIL_URL  = f"https://{HOST}/zillow/property"


def _key() -> str:
    return os.getenv("RAPIDAPI_KEY", "")


def _headers() -> dict:
    return {"x-rapidapi-host": HOST, "x-rapidapi-key": _key()}


# ── Street abbreviation normalisation ────────────────────────────────────────
_ABBREV = {
    "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr",
    "court":  "ct", "place":  "pl",  "road":      "rd",   "lane":  "ln",
    "way":    "wy", "circle": "cir", "terrace":   "ter",  "trail": "trl",
    "north":  "n",  "south":  "s",   "east":      "e",    "west":  "w",
}


def _normalise(addr: str) -> str:
    """Lowercase, remove punctuation, expand/collapse common abbreviations."""
    s = re.sub(r"[^a-z0-9 ]", "", addr.lower()).strip()
    tokens = [_ABBREV.get(w, w) for w in s.split()]
    return " ".join(tokens)


def _street_number(addr: str) -> str | None:
    m = re.match(r"^(\d+)", _normalise(addr))
    return m.group(1) if m else None


def _addr_match(input_addr: str, candidate_addr: str) -> bool:
    """
    True when both addresses have the same street number AND share ≥ 1 street-
    name token.  Very permissive — handles abbreviated vs full street names,
    missing suffix ("Lane" vs "Ln"), different ordering, etc.
    """
    a = _normalise(input_addr)
    b = _normalise(candidate_addr)

    na, nb = _street_number(a), _street_number(b)
    if not na or not nb or na != nb:
        return False

    # Non-numeric tokens
    wa = set(a.split()) - {na}
    wb = set(b.split()) - {nb}
    if not wa or not wb:
        return False

    # At least one meaningful street-name word in common
    return bool(wa & wb)


# ── Low-level API calls ───────────────────────────────────────────────────────

def _listings_from_response(data: dict) -> list[dict]:
    """Extract the listings list from any known response shape."""
    inner = data.get("data", {})
    if isinstance(inner, dict):
        return (inner.get("listings") or inner.get("results")
                or inner.get("properties") or [])
    return (data.get("listings") or data.get("results")
            or data.get("props") or [])


def _zip_search(zip_code: str, listing_type: str,
                home_type: str | None = "house",
                page: int = 1) -> list[dict]:
    """Search by zip; return raw listing dicts (empty on error)."""
    params: dict = {
        "zipcode":      str(zip_code).zfill(5),
        "listing_type": listing_type,
        "page":         page,
        "sort":         "newest",
    }
    if home_type:
        params["home_type"] = home_type
    try:
        r = requests.get(SEARCH_URL, headers=_headers(), params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(30)
            r = requests.get(SEARCH_URL, headers=_headers(), params=params, timeout=15)
        if r.status_code != 200:
            return []
        return _listings_from_response(r.json())
    except Exception as exc:
        log.debug(f"_zip_search error: {exc}")
        return []


def _text_search(address: str, city: str, state: str, zip_code: str) -> list[dict]:
    """
    Try the free-text search endpoint with a full address query.
    The endpoint requires a "location" parameter (not "q").
    Returns raw listing dicts; empty if the endpoint doesn't exist or errors.
    """
    query = f"{address}, {city}, {state} {zip_code}"
    try:
        r = requests.get(TEXT_URL, headers=_headers(),
                         params={"location": query}, timeout=15)
        if r.status_code not in (200, 201):
            log.debug(f"_text_search HTTP {r.status_code}: {r.text[:200]}")
            return []
        return _listings_from_response(r.json())
    except Exception as exc:
        log.debug(f"_text_search error: {exc}")
        return []


def _get_detail(zpid: str) -> dict | None:
    """Full property detail for a zpid; None on error."""
    try:
        r = requests.get(f"{DETAIL_URL}/{zpid}",
                         headers=_headers(), timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        return data["data"] if isinstance(data.get("data"), dict) else data
    except Exception as exc:
        log.debug(f"_get_detail error: {exc}")
        return None


def _school_rating(detail: dict) -> float | None:
    schools = detail.get("schools") or detail.get("nearby_schools") or []
    if not isinstance(schools, list):
        return None
    ratings = [s["rating"] for s in schools
               if isinstance(s, dict) and s.get("rating")]
    return round(sum(ratings) / len(ratings), 1) if ratings else None


def _zestimate_from_detail(detail: dict) -> float | None:
    raw = detail.get("zestimate")
    if not raw:
        return None
    try:
        val = float(str(raw).replace(",", ""))
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


# ── Match helper ──────────────────────────────────────────────────────────────

def _match_and_fetch(listings: list[dict],
                     target_addr: str) -> dict | None:
    """
    Walk listings, find an address match, fetch detail, return result dict.
    Returns None if nothing matches or no usable Zestimate found.
    """
    for item in listings:
        if not isinstance(item, dict):
            continue

        zpid = (item.get("zpid") or item.get("id")
                or item.get("propertyId") or item.get("zillow_id"))
        if not zpid:
            continue

        raw_addr = item.get("address") or ""
        if isinstance(raw_addr, dict):
            raw_addr = (raw_addr.get("streetAddress")
                        or raw_addr.get("street", ""))
        if not raw_addr:
            continue

        if not _addr_match(target_addr, str(raw_addr)):
            continue

        log.debug(f"Zillow matched '{raw_addr}' (zpid={zpid})")
        time.sleep(0.5)
        detail = _get_detail(str(zpid))
        if not detail:
            continue

        zest = _zestimate_from_detail(detail)
        if not zest:
            # Try the zestimate field directly from the listing item itself
            zest = _zestimate_from_detail(item)
        if not zest:
            log.debug(f"Zillow: zpid {zpid} matched but no Zestimate")
            continue

        return {"zestimate": zest, "school_rating": _school_rating(detail)}

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_zillow_data(address: str,
                    city: str,
                    state: str,
                    zip_code: str,
                    **_kwargs) -> dict | None:
    """
    Look up a property on Zillow and return its Zestimate + school_rating.

    Returns
    -------
    dict  {"zestimate": float, "school_rating": float | None}
    None  if the property cannot be found or any error occurs
    """
    if not address or not zip_code:
        return None
    if not _key():
        log.debug("RAPIDAPI_KEY not set — skipping Zillow lookup")
        return None

    target = address.strip()

    # ── Pass 1: text-based address search ────────────────────────────────
    listings = _text_search(address, city, state, zip_code)
    if listings:
        result = _match_and_fetch(listings, target)
        if result:
            return result
        time.sleep(0.3)

    # ── Pass 2: zip search, sold, house only ─────────────────────────────
    for page in (1, 2):
        listings = _zip_search(zip_code, "sold", home_type="house", page=page)
        result = _match_and_fetch(listings, target)
        if result:
            return result
        if listings:
            time.sleep(0.3)

    # ── Pass 3: zip search, for_sale, house only ─────────────────────────
    listings = _zip_search(zip_code, "for_sale", home_type="house")
    result = _match_and_fetch(listings, target)
    if result:
        return result
    if listings:
        time.sleep(0.3)

    # ── Pass 4: zip search, no home_type filter (catches condos etc.) ────
    for ltype in ("sold", "for_sale"):
        for page in (1, 2):
            listings = _zip_search(zip_code, ltype, home_type=None, page=page)
            result = _match_and_fetch(listings, target)
            if result:
                return result
            if listings:
                time.sleep(0.3)

    log.debug(f"Zillow: exhausted all search passes for '{target}' zip={zip_code}")
    return None
