"""
crag_utils.py
--------------
Pure helper functions for Corrective RAG (CRAG) on top of the existing
Weaviate-based mobile phone retrieval pipeline.

No Weaviate / FastAPI / SentenceTransformer dependency here on purpose —
this module only knows how to:
  1. Extract structured shopping constraints from a free-text query (via an
     existing local LLM, e.g. qwen2.5:0.5b).
  2. Parse messy phone attribute strings (price/RAM/battery/year) into
     numbers.
  3. Grade whether a given phone satisfies the extracted constraints.

main.py is responsible for orchestrating *retrieval* (calling Weaviate) and
deciding what corrective action to take based on the grades returned here.
"""

import json
import re

import ollama

# Reuse one of the already-pulled small models for constraint extraction.
# This must be one of the models your app already ensures are pulled
# (see ensure_models_available() in main.py) so no extra pull is needed.
CONSTRAINT_MODEL = "qwen2.5:0.5b"

CONSTRAINT_PROMPT = """Extract structured shopping constraints from a smartphone query.

Return ONLY a JSON object, nothing else (no markdown fences, no explanation),
with exactly these keys:

{{
  "max_price": <number in INR, or null>,
  "min_price": <number in INR, or null>,
  "min_ram_gb": <number, or null>,
  "min_battery_mah": <number, or null>,
  "brand": <string, or null>,
  "min_year": <number, or null>
}}

Rules:
- If a constraint is not mentioned or not implied, use null. Do not guess.
- "under X" / "below X" / "less than X" / "within X" -> max_price.
- "above X" / "over X" / "at least X" -> min_price (for money) or the
  matching min_* field (for ram/battery).
- "k" suffix means thousand: "20k" -> 20000, "15k budget" -> max_price 15000.
- Vague words like "cheap", "budget", "flagship", "good camera" with no
  number given do NOT set a numeric field. Leave it null.
- "brand" should be the literal brand/company name if mentioned
  (e.g. "Samsung", "Apple", "Xiaomi"), else null.

User Query: {query}

JSON:"""


_CONSTRAINT_KEYS = (
    "max_price", "min_price", "min_ram_gb",
    "min_battery_mah", "brand", "min_year",
)

# Common phone brand keywords -> canonical company name, used by the regex
# fast-path. Add to this list if your dataset has brands not covered here.
_BRAND_KEYWORDS = {
    "samsung": "Samsung", "apple": "Apple", "iphone": "Apple",
    "xiaomi": "Xiaomi", "redmi": "Xiaomi", "poco": "Poco",
    "oneplus": "OnePlus", "vivo": "Vivo", "oppo": "Oppo",
    "realme": "Realme", "motorola": "Motorola", "moto": "Motorola",
    "nokia": "Nokia", "honor": "Honor", "iqoo": "iQOO",
    "google": "Google", "pixel": "Google", "asus": "Asus",
    "lenovo": "Lenovo", "infinix": "Infinix", "tecno": "Tecno",
    "itel": "itel", "lg": "LG", "sony": "Sony", "huawei": "Huawei",
    "nothing": "Nothing", "micromax": "Micromax",
}

_NUM = r"(\d[\d,]*\.?\d*\s*k?)"


def _to_number(raw: str) -> float:
    raw = raw.strip().replace(",", "").replace(" ", "")
    if raw.lower().endswith("k"):
        return float(raw[:-1]) * 1000
    return float(raw)


def _extract_constraints_regex(query: str) -> dict:
    """
    Deterministic fast-path for the clearly-worded, common cases:
    'under 20000', '8gb ram', '5000mah battery', 'samsung phone', etc.

    This exists because the LLM extractor below runs on a 0.5B model —
    fast, but small enough to occasionally fail at emitting clean JSON.
    Anything explicit enough to match these patterns is resolved here
    without depending on the LLM at all; the LLM only fills in whatever
    this misses (more conversational/implicit phrasing).
    """
    q = query.lower()
    out = {key: None for key in _CONSTRAINT_KEYS}

    m = re.search(r"(\d{1,2})\s*gb\b", q)
    if m:
        out["min_ram_gb"] = float(m.group(1))

    m = re.search(r"(\d{3,5})\s*mah\b", q)
    if m:
        out["min_battery_mah"] = float(m.group(1))

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        out["min_year"] = float(m.group(1))

    m = re.search(
        r"(?:under|below|less than|within|upto|up to)\s*(?:rs\.?|inr|₹)?\s*" + _NUM,
        q,
    )
    if m:
        out["max_price"] = _to_number(m.group(1))

    m = re.search(
        r"(?:above|over|more than|at least|minimum)\s*(?:rs\.?|inr|₹)?\s*"
        + _NUM
        + r"(?!\s*gb)(?!\s*mah)",
        q,
    )
    if m:
        out["min_price"] = _to_number(m.group(1))

    for keyword, brand in _BRAND_KEYWORDS.items():
        if re.search(r"\b" + keyword + r"\b", q):
            out["brand"] = brand
            break

    return out


def _extract_constraints_llm(query: str) -> dict:
    """
    Calls a small local LLM to turn a free-text query into structured
    constraints. Falls back to "no constraints" on any parsing failure
    (bad JSON, model error, etc.) — this is the layer that's allowed to
    fail quietly, since the regex pass above already caught the clear-cut
    cases.
    """
    defaults = {key: None for key in _CONSTRAINT_KEYS}

    try:
        response = ollama.chat(
            model=CONSTRAINT_MODEL,
            messages=[
                {"role": "user", "content": CONSTRAINT_PROMPT.format(query=query)}
            ],
            options={"temperature": 0, "num_predict": 120},
        )
        raw = response["message"]["content"].strip()

        # Strip common LLM formatting noise around JSON.
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

        # In case the model adds stray text before/after the object,
        # grab the outermost {...} block.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]

        parsed = json.loads(raw)
        defaults.update({k: v for k, v in parsed.items() if k in defaults})

    except Exception:
        pass

    return defaults


def extract_constraints(query: str) -> dict:
    """
    Combines the deterministic regex pass with the LLM pass: regex wins
    whenever it found something (it's never wrong about an explicit
    "under 20000"), and the LLM only fills in fields regex left null.
    """
    regex_constraints = _extract_constraints_regex(query)
    llm_constraints = _extract_constraints_llm(query)

    merged = {}
    for key in _CONSTRAINT_KEYS:
        merged[key] = (
            regex_constraints[key]
            if regex_constraints[key] is not None
            else llm_constraints.get(key)
        )
    return merged


def has_any_constraint(constraints: dict) -> bool:
    return any(v is not None for v in constraints.values())


def _extract_numbers(text) -> list:
    """Pulls every numeric value out of a messy string like '₹19,999'
    or '6GB/8GB' or '5000 mAh'."""
    if text is None:
        return []
    cleaned = str(text).replace(",", "")
    return [float(n) for n in re.findall(r"\d+\.?\d*", cleaned)]


def parse_price_inr(price_str):
    """If a range/multiple values are present, use the cheapest variant —
    that's the most charitable reading for a price ceiling check."""
    nums = _extract_numbers(price_str)
    return min(nums) if nums else None


def parse_ram_gb(ram_str):
    """If multiple RAM variants are listed (e.g. '4/6/8GB'), use the
    smallest — a phone only fails a min_ram_gb check if even its smallest
    variant can't meet it."""
    nums = _extract_numbers(ram_str)
    return min(nums) if nums else None


def parse_battery_mah(battery_str):
    nums = _extract_numbers(battery_str)
    return max(nums) if nums else None


def parse_year(year_str):
    nums = _extract_numbers(year_str)
    return nums[0] if nums else None


def phone_satisfies(phone: dict, constraints: dict):
    """
    Grades a single retrieved phone against extracted constraints.
    Returns (ok: bool, failed_constraints: list[str]).
    """
    failed = []

    price = parse_price_inr(phone.get("price"))
    ram = parse_ram_gb(phone.get("ram"))
    battery = parse_battery_mah(phone.get("battery"))
    year = parse_year(phone.get("launch_year"))

    if constraints.get("max_price") is not None and price is not None:
        if price > constraints["max_price"]:
            failed.append("max_price")

    if constraints.get("min_price") is not None and price is not None:
        if price < constraints["min_price"]:
            failed.append("min_price")

    if constraints.get("min_ram_gb") is not None and ram is not None:
        if ram < constraints["min_ram_gb"]:
            failed.append("min_ram_gb")

    if constraints.get("min_battery_mah") is not None and battery is not None:
        if battery < constraints["min_battery_mah"]:
            failed.append("min_battery_mah")

    if constraints.get("min_year") is not None and year is not None:
        if year < constraints["min_year"]:
            failed.append("min_year")

    if constraints.get("brand"):
        brand_query = str(constraints["brand"]).strip().lower()
        company = str(phone.get("company", "")).strip().lower()
        model_name = str(phone.get("model", "")).strip().lower()
        if (
            brand_query not in company
            and brand_query not in model_name
            and company not in brand_query
        ):
            failed.append("brand")

    return len(failed) == 0, failed