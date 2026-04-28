"""
collect_realty_images.py — Pull property photos from Realty In US (RapidAPI)
for multi-state CLIP calibration.

Budget:  ≤ 480 API calls (20 buffer from 500/month limit).
         2 calls per property: (1) list search to resolve property_id,
                               (2) get-photos to retrieve image URLs.
         → up to 240 properties attempted.

Source:  data/raw/zillow_scraped.csv — non-CA homes with known addresses
         and sold prices.  These are ground-truth sold prices so residuals
         will be meaningful for calibration.

Output:
  data/raw/realty_images/          downloaded JPEG files
  data/processed/realty_manifest.csv  image_path, price, bed, bath,
                                       sqft, city, state, zip_code

Usage:
    python -m src.training.collect_realty_images            # live run
    python -m src.training.collect_realty_images --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── API config ────────────────────────────────────────────────────────────────
HOST       = "realty-in-us.p.rapidapi.com"
LIST_URL   = f"https://{HOST}/properties/v3/list"
DETAIL_URL = f"https://{HOST}/properties/v3/detail"

def _headers():
    key = os.getenv("RAPIDAPI_KEY", "")
    return {
        "x-rapidapi-host": HOST,
        "x-rapidapi-key":  key,
        "Content-Type":    "application/json",
    }

# ── Budget ────────────────────────────────────────────────────────────────────
# Default budget for a re-run: 150 calls (75 property attempts).
# Override with --budget N on the command line.
# First full run used ~330-400 calls; this conservative default protects
# the remaining monthly quota (500 total, 20 buffer → 480 effective).
CALL_BUDGET      = 150          # overridden by --budget arg at runtime
CALLS_PER_PROP   = 2            # list search + photo fetch
MAX_PROPERTIES   = CALL_BUDGET // CALLS_PER_PROP   # updated in main()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRAPED_CSV   = Path("data/raw/zillow_scraped.csv")
IMAGE_DIR     = Path("data/raw/realty_images")
MANIFEST_PATH = Path("data/processed/realty_manifest.csv")

# ── State name normalisation ──────────────────────────────────────────────────
_STATE_ABBREV = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas",
    "CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware",
    "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho",
    "IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland",
    "MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi",
    "MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada",
    "NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York",
    "NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
    "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah",
    "VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
    "WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
}

def _full_state(abbrev: str) -> str:
    s = abbrev.strip().upper()
    return _STATE_ABBREV.get(s, abbrev.strip().title())


# ── Realty API helpers ────────────────────────────────────────────────────────

def _extract_results(data: dict) -> list[dict]:
    """
    Extract the results list from any known v3/list response shape.

    The API has returned at least two shapes in the wild:
      • data.home_search.results   (most common for v3)
      • data.results               (older / alternate endpoint)
    """
    inner = data.get("data") or {}
    # v3 standard: data → home_search → results
    hs = inner.get("home_search") or {}
    if hs.get("results"):
        return hs["results"]
    # fallback shapes
    return (inner.get("results")
            or inner.get("listings")
            or data.get("results")
            or [])


def _search_property(address: str, postal_code: str) -> str | None:
    """
    POST to /properties/v3/list with address + postal_code.
    Returns the first matching property_id, or None.
    """
    body = {
        "limit":       5,
        "offset":      0,
        "postal_code": str(postal_code).zfill(5),
        "address":     address,
        "status":      ["sold", "for_sale"],
        "sort":        {"direction": "desc", "field": "sold_date"},
    }
    try:
        r = requests.post(LIST_URL, headers=_headers(), json=body, timeout=15)
        if r.status_code == 429:
            log.warning("Rate-limited — sleeping 60 s")
            time.sleep(60)
            r = requests.post(LIST_URL, headers=_headers(), json=body, timeout=15)
        if r.status_code != 200:
            log.debug(f"list HTTP {r.status_code} for {address}")
            return None
        data    = r.json()
        results = _extract_results(data)
        if not results:
            return None
        first = results[0]
        pid = (first.get("property_id")
               or first.get("listing_id")
               or (first.get("advertisers") or [{}])[0].get("id"))
        return str(pid) if pid else None
    except Exception as exc:
        log.debug(f"_search_property error: {exc}")
        return None



# Interior tags that CLIP's condition prompts are designed to score.
# Photos tagged with these are far more informative than exterior shots.
_INTERIOR_TAGS = {
    "kitchen", "bathroom", "living_room", "bedroom", "dining_room",
    "family_room", "office", "laundry_room", "basement", "interior",
    "master_bedroom", "master_bathroom",
}
_EXTERIOR_TAGS = {"house_view", "porch", "yard", "pool", "garage", "aerial"}


def _rank_photos(photos: list[dict]) -> list[str]:
    """
    Return photo URLs ordered: interior first, then other, then exterior.
    Tags come free inside the detail response — no extra API calls.
    Ties broken by tag probability (highest confidence first).
    """
    interior, neutral, exterior = [], [], []

    for p in photos:
        if not isinstance(p, dict):
            continue
        href = p.get("href") or p.get("url") or p.get("src") or ""
        if not href:
            continue

        tags       = p.get("tags") or []
        tag_labels = {(t.get("label") or "").lower() for t in tags
                      if isinstance(t, dict)}
        top_prob   = max((t.get("probability", 0) for t in tags
                          if isinstance(t, dict)), default=0)

        if tag_labels & _INTERIOR_TAGS:
            interior.append((top_prob, href))
        elif tag_labels & _EXTERIOR_TAGS:
            exterior.append((top_prob, href))
        else:
            neutral.append((top_prob, href))

    def _urls(lst):
        return [href for _, href in sorted(lst, key=lambda x: -x[0])]

    return _urls(interior) + _urls(neutral) + _urls(exterior)


def _get_photos(property_id: str) -> list[str]:
    """
    GET /properties/v3/detail for a property_id and return photo URLs.

    URLs are ranked: interior photos first (kitchen, bathroom, living_room…),
    then untagged, then exterior.  All tags arrive in the same detail response
    — no extra API calls beyond the one detail fetch.

    The get-photos endpoint is not used: it ignores property_id and returns
    an empty home_search result.
    """
    try:
        r = requests.get(
            DETAIL_URL,
            headers=_headers(),
            params={"property_id": property_id},
            timeout=15,
        )
        if r.status_code == 429:
            log.warning("Rate-limited — sleeping 60 s")
            time.sleep(60)
            r = requests.get(
                DETAIL_URL,
                headers=_headers(),
                params={"property_id": property_id},
                timeout=15,
            )
        if r.status_code != 200:
            return []
        data   = r.json()
        inner  = data.get("data") or {}
        home   = inner.get("home") or {}
        photos = (inner.get("photos")
                  or home.get("photos")
                  or home.get("property_photos")
                  or [])
        return _rank_photos(photos)
    except Exception as exc:
        log.debug(f"_get_photos error: {exc}")
        return []


def _download_image(url: str, dest: Path) -> bool:
    """Download an image URL to dest. Returns True on success."""
    try:
        r = requests.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        content_type = r.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, budget: int = CALL_BUDGET):
    if not os.getenv("RAPIDAPI_KEY") and not dry_run:
        log.error("RAPIDAPI_KEY not set in .env — cannot proceed.")
        sys.exit(1)

    call_budget   = budget
    max_props     = call_budget // CALLS_PER_PROP
    log.info(f"Budget this run: {call_budget} calls → up to {max_props} new properties")

    # ── Load source data ──────────────────────────────────────────────────────
    log.info(f"Loading {SCRAPED_CSV} …")
    df = pd.read_csv(SCRAPED_CSV)
    log.info(f"  {len(df):,} total scraped rows")

    # Filter to non-CA homes with usable addresses and prices
    ca_states = {"CA", "California"}
    non_ca = df[
        ~df["state"].isin(ca_states)
        & df["address"].notna()
        & df["address"].str.strip().ne("")
        & df["zip_code"].notna()
        & df["price"].notna()
        & (df["price"] > 50_000)
        & df["sqft"].notna()
        & (df["sqft"] > 200)
    ].copy()

    log.info(f"  {len(non_ca):,} non-CA rows with address + price")

    # Normalise state to full names (structured model uses full names)
    non_ca["state_full"] = non_ca["state"].apply(_full_state)

    # Prioritise higher-value homes — these give the best signal for premium
    # market calibration (the exact gap we need to fix).
    # Sample with probability proportional to price rank so cheap homes still
    # get some representation.
    non_ca = non_ca.sort_values("price", ascending=False).reset_index(drop=True)
    n_target = min(max_props, len(non_ca))

    # Weight: top 40% by price get 3x sampling weight
    weights        = np.ones(len(non_ca))
    premium_cutoff = int(len(non_ca) * 0.40)
    weights[:premium_cutoff] = 3.0
    weights        = weights / weights.sum()

    rng      = np.random.default_rng(42)
    sampled  = non_ca.iloc[
        rng.choice(len(non_ca), size=n_target, replace=False, p=weights)
    ].reset_index(drop=True)

    log.info(f"  Targeting {n_target} properties "
             f"(budget {CALL_BUDGET} calls / {CALLS_PER_PROP} per property)")
    log.info(f"  State distribution:")
    for state, cnt in sampled["state"].value_counts().head(12).items():
        log.info(f"    {state:20s}: {cnt}")

    if dry_run:
        log.info("\nDry-run — no API calls made.  Remove --dry-run to collect images.")
        return

    # ── Load existing manifest to skip already-downloaded homes ──────────────
    done_addresses: set[str] = set()
    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        done_addresses = set(existing["address"].dropna().str.lower())
        log.info(f"  {len(done_addresses)} already in manifest — skipping")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect ───────────────────────────────────────────────────────────────
    records    = []
    calls_used = 0
    ok_count   = 0
    fail_count = 0

    for i, row in sampled.iterrows():
        if calls_used + CALLS_PER_PROP > call_budget:
            log.warning(f"Budget exhausted at {calls_used} calls — stopping.")
            break

        addr = str(row["address"]).strip()
        if addr.lower() in done_addresses:
            continue

        zip_code = str(row["zip_code"]).split(".")[0].zfill(5)
        log.info(f"[{i+1}/{n_target}] {addr}, {row['city']}, {row['state']}  "
                 f"${row['price']:,.0f}  (calls used: {calls_used})")

        # ── Call 1: resolve property_id ────────────────────────────────────
        prop_id = _search_property(addr, zip_code)
        calls_used += 1
        time.sleep(0.4)

        if not prop_id:
            log.debug(f"  No property_id found — skipping")
            fail_count += 1
            continue

        # ── Call 2: get photos ─────────────────────────────────────────────
        photo_urls = _get_photos(prop_id)
        calls_used += 1
        time.sleep(0.4)

        if not photo_urls:
            log.debug(f"  No photos for property_id={prop_id}")
            fail_count += 1
            continue

        # Download first available photo
        safe_addr = "".join(c if c.isalnum() else "_" for c in addr)[:60]
        img_path  = IMAGE_DIR / f"{safe_addr}_{zip_code}.jpg"

        downloaded = False
        for url in photo_urls[:3]:   # try first 3 in case one fails
            if _download_image(url, img_path):
                downloaded = True
                break

        if not downloaded:
            log.debug(f"  Image download failed for {addr}")
            fail_count += 1
            continue

        records.append({
            "image_path": str(img_path),
            "address":    addr,
            "city":       row["city"],
            "state":      row["state_full"],
            "zip_code":   zip_code,
            "price":      float(row["price"]),
            "bed":        float(row["bed"])  if pd.notna(row.get("bed"))  else np.nan,
            "bath":       float(row["bath"]) if pd.notna(row.get("bath")) else np.nan,
            "sqft":       float(row["sqft"]) if pd.notna(row.get("sqft")) else np.nan,
        })
        ok_count += 1
        log.info(f"  ✓ saved  ({ok_count} collected, {fail_count} failed, "
                 f"{calls_used} calls)")

    # ── Save manifest ─────────────────────────────────────────────────────────
    if records:
        new_df = pd.DataFrame(records)

        if MANIFEST_PATH.exists():
            existing = pd.read_csv(MANIFEST_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(MANIFEST_PATH, index=False)
            log.info(f"\nAppended {len(new_df)} rows → manifest now has "
                     f"{len(combined)} total rows")
        else:
            new_df.to_csv(MANIFEST_PATH, index=False)
            log.info(f"\nManifest saved: {len(new_df)} rows → {MANIFEST_PATH}")

        log.info(f"Images saved to: {IMAGE_DIR}")
    else:
        log.warning("No images collected — manifest not written.")

    log.info(f"\nSummary: {ok_count} collected | {fail_count} failed | "
             f"{calls_used}/{call_budget} calls used this run")


def debug_one(address: str, postal_code: str):
    """
    Make ONE list call for a given address and dump the full raw response.
    Use this to verify the response shape before the full run.

    Usage:
        python -m src.training.collect_realty_images \
            --debug --address "4017 Bobbin Ln" --zip 75001
    """
    body = {
        "limit":       5,
        "offset":      0,
        "postal_code": str(postal_code).zfill(5),
        "address":     address,
        "status":      ["sold", "for_sale"],
        "sort":        {"direction": "desc", "field": "sold_date"},
    }
    print(f"\nPOST {LIST_URL}")
    print(f"Body: {json.dumps(body, indent=2)}\n")
    r = requests.post(LIST_URL, headers=_headers(), json=body, timeout=15)
    print(f"HTTP {r.status_code}")
    data = r.json()
    print("Raw response (truncated to 3000 chars):")
    print(json.dumps(data, indent=2)[:3000])

    results = _extract_results(data)
    print(f"\n── Extracted {len(results)} result(s) ──")
    for i, res in enumerate(results[:3]):
        pid = (res.get("property_id")
               or res.get("listing_id")
               or (res.get("advertisers") or [{}])[0].get("id"))
        addr = ((res.get("location") or {}).get("address") or {})
        print(f"  [{i}] property_id={pid}  "
              f"addr={addr.get('line','')} {addr.get('city','')} {addr.get('state_code','')}")

    if results:
        pid = (results[0].get("property_id")
               or results[0].get("listing_id"))
        if pid:
            print(f"\n── Fetching detail for property_id={pid} ──")
            r2 = requests.get(
                DETAIL_URL,
                headers=_headers(),
                params={"property_id": str(pid)},
                timeout=15,
            )
            print(f"HTTP {r2.status_code}")
            detail = r2.json()
            # Show top-level structure
            inner  = (detail.get("data") or {})
            home   = inner.get("home") or {}
            photos = (inner.get("photos")
                      or home.get("photos")
                      or home.get("property_photos")
                      or [])
            print(f"  home keys: {list(home.keys())[:15]}")
            print(f"  photos found: {len(photos)}")
            ranked = _rank_photos(photos)
            print(f"  ranked URLs (interior-first), showing top 5:")
            for i, url in enumerate(ranked[:5]):
                print(f"    [{i}] {url[:90]}")
            if not photos:
                print("  No photos — dumping data.home (first 2000 chars):")
                print(json.dumps(home, indent=2)[:2000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sampling plan without making API calls")
    parser.add_argument("--debug",   action="store_true",
                        help="Test one address and dump raw API response (2 calls)")
    parser.add_argument("--address", default="4017 Bobbin Ln",
                        help="Address to test in --debug mode")
    parser.add_argument("--zip",     default="75001",
                        help="Zip code to test in --debug mode")
    parser.add_argument("--budget",  type=int, default=150,
                        help="Max API calls to use this run (default 150 = ~75 properties). "
                             "Check your RapidAPI dashboard for remaining monthly quota before running.")
    args = parser.parse_args()

    if args.debug:
        debug_one(args.address, args.zip)
    else:
        main(dry_run=args.dry_run, budget=args.budget)
