#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert Data2Vis Vega-Lite examples into a Table2Charts chart corpus.

Table2Charts does not train against a full Vega-Lite string.  Its chart
supervision is an action sequence over a table-specific action space:

    ANA -> VAL field(s) -> SEP -> X field(s) -> SEP/GRP

This converter therefore keeps the parts of Data2Vis that match that objective:
full source tables from ``examplesdata`` and field-level chart decisions from
``examples/*/*.vl.json``.  It intentionally skips transformations that cannot be
represented by the Table2Charts chart grammar, such as ``count(*)`` labels or
non-field marks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


RAW_TYPE_STRING = 1
RAW_TYPE_DATETIME = 3
RAW_TYPE_DECIMAL = 5
RAW_TYPE_YEAR = 7

MAJOR_MARK_TO_ANA = {
    "bar": "barChart",
    "line": "lineChart",
    "point": "scatterChart",
    "circle": "scatterChart",
    "square": "scatterChart",
}

OPTIONAL_MARK_TO_ANA = {
    "area": "areaChart",
}

MISSING = {"", "nan", "none", "null", "na", "n/a"}
VEGA_NUMERIC_TYPES = {"quantitative", "temporal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Data2Vis examples into Table2Charts corpus layout."
    )
    parser.add_argument(
        "--data2vis-root",
        type=Path,
        default=Path("Data/Data2VisRaw"),
        help="Path to the cloned victordibia/data2vis repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/Data2VisTable2Charts"),
        help="Output corpus root in Table2Charts layout.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum rows per source table used for feature extraction.",
    )
    parser.add_argument(
        "--max-specs-per-dataset",
        type=int,
        default=0,
        help="Optional cap after filtering for each Data2Vis dataset; 0 means no cap.",
    )
    parser.add_argument(
        "--include-area",
        action="store_true",
        help="Include Vega-Lite area marks as areaChart labels. Default keeps major chart types only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8-sig")


def as_mark_name(mark: Any) -> Optional[str]:
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        mark_type = mark.get("type")
        return mark_type if isinstance(mark_type, str) else None
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return str(value).strip().lower() in MISSING


def to_float(value: Any) -> Optional[float]:
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        val = float(str(value).replace(",", ""))
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def looks_like_date(value: Any) -> bool:
    if is_missing(value):
        return False
    text = str(value).strip()
    # Keep this deliberately conservative; numeric year handling is separate.
    patterns = [
        r"^\d{4}-\d{1,2}-\d{1,2}",
        r"^\d{1,2}/\d{1,2}/\d{2,4}",
        r"^\d{1,2}-\d{1,2}-\d{2,4}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def is_year_like(name: str, values: Iterable[Any]) -> bool:
    if "year" in name.lower():
        return True
    nums = [to_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return False
    plausible = [v for v in nums if abs(v - round(v)) < 1e-8 and 1000 <= v <= 2200]
    return len(plausible) / len(nums) >= 0.9


def infer_field_types(
    rows: List[Dict[str, Any]],
    fields: List[str],
    type_hints: Dict[str, Counter],
) -> Dict[str, int]:
    inferred: Dict[str, int] = {}
    for field in fields:
        values = [row.get(field, "") for row in rows]
        non_missing = [v for v in values if not is_missing(v)]
        hints = type_hints.get(field, Counter())
        if hints["temporal"] > 0:
            inferred[field] = RAW_TYPE_YEAR if is_year_like(field, values) else RAW_TYPE_DATETIME
            continue
        if hints["quantitative"] > 0:
            inferred[field] = RAW_TYPE_YEAR if is_year_like(field, values) else RAW_TYPE_DECIMAL
            continue
        if non_missing:
            numeric_ratio = sum(to_float(v) is not None for v in non_missing) / len(non_missing)
            date_ratio = sum(looks_like_date(v) for v in non_missing) / len(non_missing)
            if numeric_ratio >= 0.85:
                inferred[field] = RAW_TYPE_YEAR if is_year_like(field, values) else RAW_TYPE_DECIMAL
                continue
            if date_ratio >= 0.85:
                inferred[field] = RAW_TYPE_DATETIME
                continue
        inferred[field] = RAW_TYPE_STRING
    return inferred


def entropy(items: Iterable[str]) -> float:
    counts = Counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def string_features(values: List[Any], n_rows: int) -> Dict[str, float]:
    texts = ["" if is_missing(v) else str(v) for v in values]
    non_empty = [text for text in texts if text != ""]
    sample_count = len(texts)
    if sample_count == 0:
        sample_count = 1
    prefix = Counter(text[0] for text in non_empty if text)
    suffix = Counter(text[-1] for text in non_empty if text)
    lengths = [len(text) for text in texts]
    changed = sum(1 for i in range(1, len(texts)) if texts[i] != texts[i - 1])
    distinct = len(set(texts))
    major = max(Counter(texts).values()) / len(texts) if texts else 1.0
    return {
        "commonPrefix": max(prefix.values()) / sample_count if prefix else 0.0,
        "commonSuffix": max(suffix.values()) / sample_count if suffix else 0.0,
        "keyEntropy": entropy(texts),
        "charEntropy": entropy(char for text in texts for char in text),
        "changeRate": changed / (len(texts) - 1) if len(texts) > 1 else 0.0,
        "cardinality": distinct / len(texts) if texts else 0.0,
        "major": major,
        "medianLength": float(median(lengths)) if lengths else 0.0,
        "lengthVariance": variance(lengths),
        "averageLogLength": sum(math.log10(max(1, length)) for length in lengths) / len(lengths)
        if lengths
        else 0.0,
        "absoluteCardinality": float(distinct),
        "nRows": float(n_rows),
    }


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def stddev(values: List[float]) -> float:
    return math.sqrt(variance(values))


def benford_distance(nums: List[float]) -> float:
    expected = [0.0, 0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    digits = Counter()
    for num in nums:
        text = f"{abs(num):.12g}".lstrip("0.")
        for char in text:
            if char.isdigit() and char != "0":
                digits[int(char)] += 1
                break
    total = sum(digits.values())
    if total == 0:
        return 0.0
    return math.sqrt(sum(((digits[i] / total) - expected[i]) ** 2 for i in range(1, 10)))


def gini(nums: List[float]) -> float:
    positives = sorted(v for v in nums if v >= 0)
    if not positives:
        return 0.0
    total = sum(positives)
    if abs(total) < 1e-12:
        return 0.0
    n = len(positives)
    weighted = sum((2 * i - n - 1) * value for i, value in enumerate(positives, start=1))
    return weighted / (n * total)


def progression_confidence(nums: List[float], geometric: bool = False) -> float:
    if len(nums) <= 2:
        return 1.0 if nums else 0.0
    values = nums[:]
    if geometric:
        if any(v == 0 for v in values):
            return 0.0
        same_sign = all(v > 0 for v in values) or all(v < 0 for v in values)
        if not same_sign:
            return 0.0
        values = [math.log10(abs(v)) for v in values]
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    scale = max(1e-12, max(abs(v) for v in values))
    score = 1.0 / (1.0 + stddev(diffs) / scale)
    return max(0.0, min(1.0, score))


def numeric_features(values: List[Any], n_rows: int) -> Dict[str, float]:
    nums = [to_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    texts = ["" if is_missing(v) else str(v) for v in values]
    base = string_features(values, n_rows)
    if not nums:
        base.update(
            {
                "aggrPercentFormatted": 0.0,
                "aggr01Ranged": 0.0,
                "aggr0100Ranged": 0.0,
                "aggrIntegers": 0.0,
                "aggrNegative": 0.0,
                "range": 0.0,
                "partialOrdered": 0.0,
                "variance": 0.0,
                "cov": 0.0,
                "spread": 0.0,
                "benford": 0.0,
                "orderedConfidence": 0.0,
                "equalProgressionConfidence": 0.0,
                "geometricProgressionConfidence": 0.0,
                "sumIn01": 0.0,
                "sumIn0100": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "gini": 0.0,
            }
        )
        return base

    sorted_nums = sorted(nums)
    mean = sum(nums) / len(nums)
    sigma = stddev(nums)
    distinct = len(set(round(v, 12) for v in nums))
    data_range = sorted_nums[-1] - sorted_nums[0] if sorted_nums else 0.0
    if len(nums) > 1:
        inc = sum(1 for i in range(1, len(nums)) if nums[i] >= nums[i - 1])
        dec = sum(1 for i in range(1, len(nums)) if nums[i] <= nums[i - 1])
        changed = sum(1 for i in range(1, len(nums)) if abs(nums[i] - nums[i - 1]) > 1e-9)
        partial_ordered = max(inc, dec) / (len(nums) - 1)
        change_rate = changed / (len(nums) - 1)
    else:
        partial_ordered = change_rate = 0.0
    if sigma > 1e-12 and len(nums) > 2:
        skewness = sum((v - mean) ** 3 for v in nums) / (len(nums) * sigma**3)
        kurtosis = sum((v - mean) ** 4 for v in nums) / (len(nums) * sigma**4)
    else:
        skewness = kurtosis = 0.0

    base.update(
        {
            "aggrPercentFormatted": 0.0,
            "aggr01Ranged": sum(0 <= v <= 1 for v in nums) / len(nums),
            "aggr0100Ranged": sum(0 <= v <= 100 for v in nums) / len(nums),
            "aggrIntegers": sum(abs(v - round(v)) < 1e-8 for v in nums) / len(nums),
            "aggrNegative": sum(v < 0 for v in nums) / len(nums),
            "range": data_range,
            "changeRate": change_rate,
            "partialOrdered": partial_ordered,
            "variance": sigma,
            "cov": sigma / mean if abs(mean) > 1e-12 else 0.0,
            "cardinality": distinct / len(nums),
            "spread": (distinct / len(nums)) / (data_range + 1.0),
            "major": max(Counter(nums).values()) / len(nums),
            "benford": benford_distance(nums),
            "orderedConfidence": partial_ordered,
            "equalProgressionConfidence": progression_confidence(nums, geometric=False),
            "geometricProgressionConfidence": progression_confidence(nums, geometric=True),
            # Preserve the misspelling used by older DataConfig cleanup code.
            "geometircProgressionConfidence": progression_confidence(nums, geometric=True),
            "medianLength": float(median([len(text) for text in texts])) if texts else 0.0,
            "lengthVariance": variance([len(text) for text in texts]),
            "sumIn01": sum(nums) if 0 <= sum(nums) <= 1 else 0.0,
            "sumIn0100": sum(nums) if 0 <= sum(nums) <= 100 else 0.0,
            "absoluteCardinality": float(distinct),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "gini": gini(nums),
            "nRows": float(n_rows),
        }
    )
    return base


def cell_type_counter(raw_type: int, values: List[Any]) -> Dict[str, int]:
    counter = {"Unknown": 0, "String": 0, "DateTime": 0, "Decimal": 0, "Year": 0}
    key = {
        RAW_TYPE_STRING: "String",
        RAW_TYPE_DATETIME: "DateTime",
        RAW_TYPE_DECIMAL: "Decimal",
        RAW_TYPE_YEAR: "Year",
    }.get(raw_type, "Unknown")
    counter[key] = sum(not is_missing(v) for v in values)
    counter["Unknown"] = sum(is_missing(v) for v in values)
    return counter


def make_field(name: str, index: int, values: List[Any], raw_type: int, n_rows: int) -> Dict[str, Any]:
    features = (
        numeric_features(values, n_rows)
        if raw_type in {RAW_TYPE_DECIMAL, RAW_TYPE_YEAR}
        else string_features(values, n_rows)
    )
    null_ratio = sum(is_missing(v) for v in values) / len(values) if values else 0.0
    features["indexRatio"] = index / max(1, n_rows)
    features["nullRatio"] = null_ratio
    return {
        "name": name,
        "index": index,
        "type": raw_type,
        "dataFeatures": features,
        "cellTypeCounter": cell_type_counter(raw_type, values),
        "additionalFeatures": {
            "truncatedHeader": name[:64].lower(),
            "extractedTokens": "",
            "identifiedTokens": [],
            "isPercent": False,
            "isCurrency": False,
            "translatedHeaders": split_header(name),
            "nullRatio": null_ratio,
        },
        "inHeaderRegion": False,
        "isPercent": False,
        "isCurrency": False,
        "hasYear": raw_type == RAW_TYPE_YEAR,
        "hasMonth": False,
        "hasDay": raw_type == RAW_TYPE_DATETIME,
        "isSequence": False,
        "isOrdinal": raw_type == RAW_TYPE_YEAR,
    }


def split_header(name: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", name.replace("_", " "))
    return [token.lower() for token in tokens] or [name.lower()]


def hashed_token_vector(token: str, dim: int = 50) -> List[float]:
    vec = [0.0] * dim
    pieces = [token]
    if len(token) > 3:
        pieces.extend(token[i : i + 3] for i in range(len(token) - 2))
    for piece in pieces:
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=16).digest()
        for i in range(0, len(digest), 2):
            bucket = digest[i] % dim
            sign = 1.0 if digest[i + 1] % 2 == 0 else -1.0
            vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 8) for v in vec]


def make_embedding(name: str) -> Dict[str, Dict[str, List[Any]]]:
    tokens = split_header(name)
    origin = [hashed_token_vector(token) for token in tokens]
    mean = [round(sum(vec[i] for vec in origin) / len(origin), 8) for i in range(50)]
    return {"0": {"mean": mean, "origin": origin}}


def data_path_for_spec(spec: Dict[str, Any], data_root: Path) -> Optional[Path]:
    data = spec.get("data")
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    if not isinstance(url, str):
        return None
    filename = Path(url).name
    candidate = data_root / filename
    return candidate if candidate.exists() else None


def load_rows_for_spec(spec: Dict[str, Any], data_root: Path) -> Optional[List[Dict[str, Any]]]:
    data = spec.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        rows = data["values"]
    else:
        path = data_path_for_spec(spec, data_root)
        if path is None:
            return None
        rows = load_json(path)
    if isinstance(rows, dict):
        rows = rows.get("values") or rows.get("data")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return None
    return rows


def rows_cache_key(spec: Dict[str, Any], data_root: Path) -> Optional[str]:
    data = spec.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        # Inline examples are rare in Data2Vis; using a content hash avoids
        # accidentally merging distinct inline tables.
        payload = json.dumps(data["values"], sort_keys=True, ensure_ascii=False)
        return "inline:" + hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()
    path = data_path_for_spec(spec, data_root)
    return str(path.resolve()) if path is not None else None


def collect_fields(rows: List[Dict[str, Any]], specs: Iterable[Dict[str, Any]]) -> List[str]:
    fields: "OrderedDict[str, None]" = OrderedDict()
    for spec in specs:
        enc = spec.get("encoding") or {}
        if isinstance(enc, dict):
            for channel in ("x", "y", "color", "detail", "shape", "size"):
                item = enc.get(channel)
                if isinstance(item, dict):
                    field = item.get("field")
                    if isinstance(field, str) and field != "*":
                        fields.setdefault(field, None)
    for row in rows:
        for key in row.keys():
            fields.setdefault(str(key), None)
    return list(fields.keys())


def collect_type_hints(specs: Iterable[Dict[str, Any]]) -> Dict[str, Counter]:
    hints: Dict[str, Counter] = defaultdict(Counter)
    for spec in specs:
        enc = spec.get("encoding") or {}
        if not isinstance(enc, dict):
            continue
        for item in enc.values():
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            typ = item.get("type")
            if isinstance(field, str) and field != "*" and isinstance(typ, str):
                hints[field][typ] += 1
    return hints


def field_is_value_compatible(field: str, raw_types: Dict[str, int]) -> bool:
    return raw_types.get(field) in {RAW_TYPE_DECIMAL, RAW_TYPE_YEAR, RAW_TYPE_DATETIME}


def field_is_scatter_x_compatible(field: str, raw_types: Dict[str, int]) -> bool:
    return raw_types.get(field) in {RAW_TYPE_DECIMAL, RAW_TYPE_YEAR, RAW_TYPE_DATETIME}


def stack_grouping(enc: Dict[str, Any]) -> str:
    for channel in ("x", "y"):
        item = enc.get(channel)
        if isinstance(item, dict) and item.get("stack") not in (None, False):
            return "stacked"
    return "clustered"


def parse_chart_label(
    spec: Dict[str, Any],
    raw_types: Dict[str, int],
    mark_to_ana: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    mark = as_mark_name(spec.get("mark"))
    if mark not in mark_to_ana:
        return None, f"unsupported_mark:{mark}"
    enc = spec.get("encoding")
    if not isinstance(enc, dict):
        return None, "missing_encoding"
    x_enc = enc.get("x")
    y_enc = enc.get("y")
    if not isinstance(x_enc, dict) or not isinstance(y_enc, dict):
        return None, "missing_xy"
    x_field = x_enc.get("field")
    y_field = y_enc.get("field")
    if not isinstance(x_field, str) or not isinstance(y_field, str):
        return None, "missing_field"
    if x_field == "*" or y_field == "*":
        return None, "count_star"

    ana_type = mark_to_ana[mark]
    if ana_type == "scatterChart":
        if not (field_is_scatter_x_compatible(x_field, raw_types) and field_is_value_compatible(y_field, raw_types)):
            return None, "scatter_needs_numeric_xy"
        value_field = y_field
        x_axis_field = x_field
    elif field_is_value_compatible(y_field, raw_types):
        value_field = y_field
        x_axis_field = x_field
    elif ana_type == "barChart" and field_is_value_compatible(x_field, raw_types):
        # Vega-Lite horizontal bars encode the measure on x and category on y.
        value_field = x_field
        x_axis_field = y_field
    else:
        return None, "no_value_field"

    chart = {
        "anaType": ana_type,
        "xFieldName": x_axis_field,
        "yFieldName": value_field,
        "nVals": 1,
        "sourceMark": mark,
    }
    if ana_type == "barChart":
        chart["grouping"] = stack_grouping(enc)
    return chart, None


def schema_id_from_dataset(dataset_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", dataset_name.removesuffix("1"))


def write_dataset_schema(
    dataset_name: str,
    rows: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    out_root: Path,
    max_rows: int,
    mark_to_ana: Dict[str, str],
    max_specs: int,
) -> Tuple[Optional[str], Counter]:
    stats = Counter()
    rows = rows[:max_rows] if max_rows > 0 else rows
    if not rows:
        stats["empty_rows"] += 1
        return None, stats

    fields = collect_fields(rows, specs)
    if not fields:
        stats["empty_fields"] += 1
        return None, stats
    type_hints = collect_type_hints(specs)
    raw_types = infer_field_types(rows, fields, type_hints)
    field_index = {field: i for i, field in enumerate(fields)}

    chart_labels = []
    skip_reasons = Counter()
    for spec in specs:
        chart, reason = parse_chart_label(spec, raw_types, mark_to_ana)
        if chart is None:
            skip_reasons[reason or "unknown"] += 1
            continue
        if chart["xFieldName"] not in field_index or chart["yFieldName"] not in field_index:
            skip_reasons["field_not_in_table"] += 1
            continue
        chart_labels.append(chart)
        if max_specs and len(chart_labels) >= max_specs:
            break

    if not chart_labels:
        stats.update(skip_reasons)
        stats["no_convertible_charts"] += 1
        return None, stats

    schema_id = schema_id_from_dataset(dataset_name)
    t_uid = f"{schema_id}.t0"
    c_uids = [f"{t_uid}.c{i}" for i in range(len(chart_labels))]
    c_types = [chart["anaType"] for chart in chart_labels]

    table_fields = []
    for idx, field in enumerate(fields):
        values = [row.get(field, "") for row in rows]
        table_fields.append(make_field(field, idx, values, raw_types[field], len(rows)))

    df_json = {
        "tUid": t_uid,
        "pUids": [],
        "cUids": c_uids,
        "isExternal": False,
        "nColumns": len(fields),
        "nRows": len(rows),
        "fields": table_fields,
    }
    dump_json(df_json, out_root / "data" / f"{t_uid}.DF.json")

    sample_items = []
    for i, chart in enumerate(chart_labels):
        x_idx = field_index[chart["xFieldName"]]
        y_idx = field_index[chart["yFieldName"]]
        chart_json = {
            "xFields": [{"index": x_idx}],
            "yFields": [{"index": y_idx}],
        }
        if chart["anaType"] == "barChart":
            chart_json["grouping"] = chart.get("grouping", "clustered")
        dump_json(chart_json, out_root / "data" / f"{t_uid}.c{i}.json")
        sample_items.append({"anaType": chart["anaType"], "nVals": 1, "index": str(i)})

    sample_json = {
        "sID": schema_id,
        "lang": "en",
        "nColumns": len(fields),
        "tableAnalysisPairs": {"0": sample_items},
    }
    dump_json(sample_json, out_root / "sample-new" / f"{schema_id}.sample.json")

    embeddings = [make_embedding(field) for field in fields]
    dump_json(embeddings, out_root / "embeddings" / "fasttext" / f"{schema_id}.EMB.json")

    stats["schemas"] += 1
    stats["charts"] += len(chart_labels)
    stats["fields"] += len(fields)
    stats["rows"] += len(rows)
    stats.update({f"skipped:{k}": v for k, v in skip_reasons.items()})
    stats.update({f"ana:{ana}": c_types.count(ana) for ana in sorted(set(c_types))})
    return schema_id, stats


def grouped_specs(examples_root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(examples_root.glob("*/*.json")):
        groups[path.parent.name].append(path)
    return groups


def main() -> int:
    args = parse_args()
    data2vis_root = args.data2vis_root
    examples_root = data2vis_root / "examples"
    examples_data_root = data2vis_root / "examplesdata"
    if not examples_root.exists() or not examples_data_root.exists():
        raise SystemExit(f"Data2Vis examples not found under {data2vis_root}")

    out_root = args.output
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_root} exists; pass --overwrite to replace it")
        shutil.rmtree(out_root)
    for subdir in ("data", "sample-new", "index", "embeddings/fasttext"):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)

    mark_to_ana = dict(MAJOR_MARK_TO_ANA)
    if args.include_area:
        mark_to_ana.update(OPTIONAL_MARK_TO_ANA)

    schema_ids: List[str] = []
    report = {
        "data2vis_root": str(data2vis_root),
        "output": str(out_root),
        "max_rows": args.max_rows,
        "include_area": args.include_area,
        "datasets": {},
        "totals": Counter(),
    }

    for dataset_name, spec_paths in grouped_specs(examples_root).items():
        specs = []
        rows = None
        rows_cache: Dict[str, List[Dict[str, Any]]] = {}
        load_errors = Counter()
        for spec_path in spec_paths:
            try:
                spec = load_json(spec_path)
                key = rows_cache_key(spec, examples_data_root)
                if key is None:
                    spec_rows = None
                elif key in rows_cache:
                    spec_rows = rows_cache[key]
                else:
                    spec_rows = load_rows_for_spec(spec, examples_data_root)
                    if spec_rows is not None:
                        rows_cache[key] = spec_rows
            except Exception:
                load_errors["read_error"] += 1
                continue
            if spec_rows is None:
                load_errors["missing_rows"] += 1
                continue
            if rows is None:
                rows = spec_rows
            specs.append(spec)
        if rows is None:
            report["datasets"][dataset_name] = {"load_errors": dict(load_errors)}
            report["totals"].update(load_errors)
            continue
        schema_id, stats = write_dataset_schema(
            dataset_name=dataset_name,
            rows=rows,
            specs=specs,
            out_root=out_root,
            max_rows=args.max_rows,
            mark_to_ana=mark_to_ana,
            max_specs=args.max_specs_per_dataset,
        )
        stats.update({f"load:{k}": v for k, v in load_errors.items()})
        report["datasets"][dataset_name] = dict(stats)
        report["totals"].update(stats)
        if schema_id is not None:
            schema_ids.append(schema_id)
            print(f"{dataset_name}: {stats.get('charts', 0)} charts -> {schema_id}", flush=True)
        else:
            print(f"{dataset_name}: no convertible charts", flush=True)

    dump_json(schema_ids, out_root / "index" / "schema_ids.json")
    report["schema_ids"] = schema_ids
    report["totals"] = dict(report["totals"])
    dump_json(report, out_root / "conversion_report.json")

    print(f"Wrote {len(schema_ids)} schemas to {out_root}")
    print(f"Charts: {report['totals'].get('charts', 0)}")
    for key, value in sorted(report["totals"].items()):
        if key.startswith("ana:"):
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
