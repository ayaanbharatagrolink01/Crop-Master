
"""Import the orders export into the privacy-preserving farmer profile DB.

Usage:
    python -m app.import_orders ../data/orders.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime

from .auth import mobile_hash, normalize_mobile
from .farmers import init_db, upsert_profile


EXPECTED_COLUMNS = 56
CART_DATA_INDEX = 39

# Fields after cart_data.
CART_DATA_TAIL_COUNT = EXPECTED_COLUMNS - CART_DATA_INDEX - 1


def clean_value(value) -> str:
    """Safely normalize a CSV value that may be None."""
    return str(value or "").strip()


# Placeholder tokens the export uses for missing values.
MISSING_TOKENS = {"null", "none", "nan", "n/a", "na", "-"}


def clean_field(value) -> str:
    """Like clean_value, but maps missing-value placeholders ("NULL" etc.) to ""."""
    text = clean_value(value)
    return "" if text.lower() in MISSING_TOKENS else text


def parse_dt(value):
    """Parse an order date safely."""
    text = clean_value(value)

    if not text:
        return datetime.min

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
    )

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except Exception:
        pass

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass

    return datetime.min


def repair_csv_row(
    row: list[str],
    header: list[str],
) -> dict[str, str | None]:
    """
    Repair rows where cart_data JSON was split into multiple CSV columns.

    The export contains cart_data like:

        [{"prod_id":"...",
          "attr_sku":"...",
          "vendor_id":"...",
          ...
        }]

    The quotes inside cart_data are not escaped according to CSV rules,
    so csv.reader splits the cart_data field into multiple columns.

    Fields before cart_data are stable, and there are exactly 16 fields
    after cart_data. Therefore everything between column 39 and the final
    16 columns belongs to cart_data.
    """

    header_len = len(header)

    if len(row) == header_len:
        return dict(zip(header, row))

    # If somehow fewer columns exist, pad with None.
    if len(row) < header_len:
        repaired = row + [None] * (header_len - len(row))
        return dict(zip(header, repaired))

    tail_count = header_len - CART_DATA_INDEX - 1

    prefix = row[:CART_DATA_INDEX]

    cart_parts = row[
        CART_DATA_INDEX : len(row) - tail_count
    ]

    tail = row[
        len(row) - tail_count :
    ]

    # Reconstruct the broken cart_data.
    cart_data = ",".join(
        clean_value(part)
        for part in cart_parts
    )

    repaired = (
        prefix
        + [cart_data]
        + tail
    )

    if len(repaired) < header_len:
        repaired += [None] * (
            header_len - len(repaired)
        )

    return dict(
        zip(
            header,
            repaired[:header_len],
        )
    )


def extract_product_names_from_broken_cart(
    text: str,
) -> list[str]:
    """
    Extract product names from the malformed cart_data produced by the
    orders CSV export.

    After csv.reader splits the malformed JSON, it can look like:

        [{prod_id:"ABC",
          attr_sku:"Katyayani-EMA-5-Emamectin-Benzoate-5-SG",
          vendor_id:"XYZ",
          qty:"5"}]

    The quotes around JSON keys may already have been removed by the CSV
    parser, so this parser deliberately accepts both quoted and unquoted
    keys/values.
    """

    names: list[str] = []

    patterns = (
        r"""
        ["']?attr_sku["']?
        \s*[:=]\s*
        ["']?
        ([^"',}\]]+)
        """,
        r"""
        ["']?product_name["']?
        \s*[:=]\s*
        ["']?
        ([^"',}\]]+)
        """,
        r"""
        ["']?seed_name["']?
        \s*[:=]\s*
        ["']?
        ([^"',}\]]+)
        """,
        r"""
        ["']?product["']?
        \s*[:=]\s*
        ["']?
        ([^"',}\]]+)
        """,
        r"""
        ["']?title["']?
        \s*[:=]\s*
        ["']?
        ([^"',}\]]+)
        """,
    )

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        for match in matches:
            name = clean_value(match)

            # Remove accidental trailing JSON punctuation.
            name = name.rstrip("]}")

            if name and name not in names:
                names.append(name)

    return names


def parse_cart_products(
    raw,
) -> tuple[list[str], bool]:
    """
    Parse cart_data and return:

        ([product names], parse_failed)

    Supports:
    - valid JSON list
    - valid JSON object
    - JSON stored inside a string
    - malformed JSON from the CSV export
    - plain text fallback
    """

    text = clean_value(raw)

    if not text:
        return [], False

    # ---------------------------------------------------------
    # 1. Normal JSON.
    # ---------------------------------------------------------
    try:
        obj = json.loads(text)

        if isinstance(obj, str):
            return parse_cart_products(obj)

        if isinstance(obj, dict):
            obj = [obj]

        if isinstance(obj, list):
            names: list[str] = []

            for item in obj:

                if isinstance(item, str):
                    value = clean_value(item)

                    if value:
                        names.append(value)

                elif isinstance(item, dict):

                    for key in (
                        "attr_sku",
                        "product_name",
                        "product",
                        "name",
                        "title",
                        "seed_name",
                    ):
                        value = item.get(key)

                        if value:
                            value = clean_value(value)

                            if value:
                                names.append(value)

                            break

            return (
                list(dict.fromkeys(names)),
                False,
            )

        return [], True

    except Exception:
        pass

    # ---------------------------------------------------------
    # 2. Malformed JSON fallback.
    # ---------------------------------------------------------
    names = extract_product_names_from_broken_cart(
        text
    )

    if names:
        return (
            list(dict.fromkeys(names)),
            False,
        )

    # ---------------------------------------------------------
    # 3. Plain text fallback.
    # ---------------------------------------------------------
    if text and not text.startswith(
        ("[", "{")
    ):
        return [text], False

    return [], True


def main(path: str):
    init_db()

    rows = 0
    malformed_rows = 0
    bad_mobile = 0
    bad_cart = 0

    groups = {}

    # =========================================================
    # READ CSV
    # =========================================================

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        reader = csv.reader(f)

        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(
                "CSV file is empty."
            )

        header = [
            clean_value(column)
            for column in header
        ]

        if len(header) != EXPECTED_COLUMNS:
            print(
                f"warning: expected "
                f"{EXPECTED_COLUMNS} columns, "
                f"found {len(header)}"
            )

        for raw_row in reader:

            if not raw_row:
                continue

            rows += 1

            if len(raw_row) != len(header):
                malformed_rows += 1

            row = repair_csv_row(
                raw_row,
                header,
            )

            # -------------------------------------------------
            # MOBILE
            # -------------------------------------------------

            try:
                mobile = normalize_mobile(
                    row.get("mobile")
                )
            except (
                ValueError,
                TypeError,
            ):
                bad_mobile += 1
                continue

            # -------------------------------------------------
            # PRIVACY-PRESERVING ID
            # -------------------------------------------------

            mobile_hash_value = mobile_hash(
                mobile
            )

            # -------------------------------------------------
            # DATE
            # -------------------------------------------------

            dt = parse_dt(
                row.get("create_date")
            )

            # -------------------------------------------------
            # CART PRODUCTS
            # -------------------------------------------------

            products, failed = parse_cart_products(
                row.get("cart_data")
            )

            if failed:
                bad_cart += 1

            # -------------------------------------------------
            # GROUP CUSTOMER ORDERS
            # -------------------------------------------------

            group = groups.setdefault(
                mobile_hash_value,
                {
                    "mobile": mobile,
                    "orders": [],
                },
            )

            group["orders"].append(
                (
                    dt,
                    row,
                    products,
                )
            )

    # =========================================================
    # BUILD PROFILES
    # =========================================================

    for mobile_hash_value, group in groups.items():

        orders = sorted(
            group["orders"],
            key=lambda item: item[0],
        )

        if not orders:
            continue

        newest = orders[-1][1]

        crops: list[str] = []
        soils: list[str] = []
        waters: list[str] = []
        seasons: list[str] = []

        products: list[
            tuple[str, str]
        ] = []

        # -----------------------------------------------------
        # AGGREGATE ORDERS
        # -----------------------------------------------------

        for dt, row, parsed_products in orders:

            crop_value = clean_field(
                row.get("crop")
            )

            soil_value = clean_field(
                row.get("soil")
            )

            water_value = clean_field(
                row.get("water")
            )

            season_value = clean_field(
                row.get("season")
            )

            # Crops
            if crop_value:

                crops.extend(
                    clean_field(value)
                    for value in crop_value.split(",")
                    if clean_field(value)
                )

            # Soil
            if soil_value:
                soils.append(
                    soil_value
                )

            # Water
            if water_value:
                waters.append(
                    water_value
                )

            # Season
            if season_value:
                seasons.append(
                    season_value
                )

            # Products
            for product_name in parsed_products:

                product_date = (
                    dt.date().isoformat()
                    if dt != datetime.min
                    else ""
                )

                products.append(
                    (
                        product_date,
                        product_name,
                    )
                )

        # -----------------------------------------------------
        # LOCATION
        # -----------------------------------------------------

        lat = clean_field(
            newest.get("lat")
        )

        lng = clean_field(
            newest.get("lng")
        )

        pincode = clean_field(
            newest.get("pincode")
        )

        city = clean_field(
            newest.get("city")
        )

        state = clean_field(
            newest.get("state")
        )

        # Current requested fallback:
        # district = city unless better data exists.
        district = city

        # -----------------------------------------------------
        # CROPS
        # -----------------------------------------------------

        unique_crops = list(
            dict.fromkeys(
                crop
                for crop in crops
                if crop
            )
        )

        # -----------------------------------------------------
        # MOST COMMON SOIL
        # -----------------------------------------------------

        soil = (
            Counter(soils).most_common(1)[0][0]
            if soils
            else ""
        )

        # -----------------------------------------------------
        # MOST COMMON WATER
        # -----------------------------------------------------

        water = (
            Counter(waters).most_common(1)[0][0]
            if waters
            else ""
        )

        # -----------------------------------------------------
        # MOST COMMON SEASON
        # -----------------------------------------------------

        season = (
            Counter(seasons).most_common(1)[0][0]
            if seasons
            else ""
        )

        # -----------------------------------------------------
        # RECENT PRODUCTS
        # -----------------------------------------------------

        products_sorted = sorted(
            products,
            key=lambda item: item[0],
            reverse=True,
        )

        recent_products = [
            {
                "name": name,
                "date": product_date,
            }
            for product_date, name
            in products_sorted[:5]
        ]

        # -----------------------------------------------------
        # LAST ORDER DATE
        # -----------------------------------------------------

        last_order_date = (
            orders[-1][0].date().isoformat()
            if orders[-1][0] != datetime.min
            else None
        )

        # -----------------------------------------------------
        # PROFILE
        # -----------------------------------------------------

        profile = {
            "lat": lat or None,
            "lng": lng or None,
            "pincode": pincode or None,
            "city": city or None,
            "district": district or None,
            "state": state or None,
            "crops": ", ".join(
                unique_crops
            ) or None,
            "soil": soil or None,
            "water": water or None,
            "season": season or None,
            "recent_products": recent_products,
            "last_order_date": last_order_date,
        }

        upsert_profile(
            mobile_hash_value,
            profile,
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    profiles_with_location = sum(
        bool(
            clean_field(
                group["orders"][-1][1].get("lat")
            )
            or clean_field(
                group["orders"][-1][1].get("lng")
            )
            or clean_field(
                group["orders"][-1][1].get("pincode")
            )
            or clean_field(
                group["orders"][-1][1].get("city")
            )
        )
        for group in groups.values()
    )

    profiles_with_crop = sum(
        any(
            clean_field(
                row.get("crop")
            )
            for _, row, _ in group["orders"]
        )
        for group in groups.values()
    )

    profiles_with_products = sum(
        any(
            parsed_products
            for _, _, parsed_products
            in group["orders"]
        )
        for group in groups.values()
    )

    print(
        f"rows read: {rows}"
    )

    print(
        f"customers: {len(groups)}"
    )

    print(
        f"profiles with location: "
        f"{profiles_with_location}"
    )

    print(
        f"profiles with crop: "
        f"{profiles_with_crop}"
    )

    print(
        f"profiles with recent products: "
        f"{profiles_with_products}"
    )

    print(
        f"cart_data parse failures/skips: "
        f"{bad_cart}"
    )

    print(
        f"invalid/unsupported mobile rows skipped: "
        f"{bad_mobile}"
    )

    print(
        f"malformed CSV rows repaired: "
        f"{malformed_rows}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m app.import_orders "
            "../data/orders.csv"
        )

    main(sys.argv[1])
