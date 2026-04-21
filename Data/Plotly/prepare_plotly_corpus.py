#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prepare a VizML Plotly TSV dump for Table2Charts training.

The public VizML Plotly corpus is stored as a TSV with four columns:

    fid, chart_data, layout, table_data

This script keeps the same high-level processing intent as the original VizML
and Table2Charts data notes, but avoids the old pandas/C# toolchain so it can
run in the lightweight project environment:

1. Create a VizML-compatible ``plot_data.tsv`` entry.
2. Filter rows without both chart data and table data.
3. Deduplicate near-identical tables using the simplified table signature from
   VizML's ``data_cleaning/deduplicate_charts.py``.
4. Convert the remaining Plotly table/chart pairs into the Table2Charts corpus
   layout: ``data/``, ``sample-new/``, ``embeddings/fasttext/`` and
   ``index/schema_ids.json``.

The converter intentionally emits the simplified chart JSON used by the Python
training loader. It also writes raw compatibility artifacts under
``raw/data_origin`` for inspection and for the documented ChartSplit shape.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA2VIS_CONVERTER = REPO_ROOT / "Data" / "Data2Vis" / "convert_to_table2charts.py"


def _load_data2vis_helpers():
    spec = importlib.util.spec_from_file_location("_data2vis_t2c_helpers", DATA2VIS_CONVERTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper functions from {DATA2VIS_CONVERTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_data2vis_helpers()
RAW_TYPE_STRING = HELPERS.RAW_TYPE_STRING
RAW_TYPE_DATETIME = HELPERS.RAW_TYPE_DATETIME
RAW_TYPE_DECIMAL = HELPERS.RAW_TYPE_DECIMAL
RAW_TYPE_YEAR = HELPERS.RAW_TYPE_YEAR


TSV_HEADERS = ["fid", "chart_data", "layout", "table_data"]
SUPPORTED_ANA = {"barChart", "lineChart", "scatterChart", "pieChart"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a VizML Plotly TSV dump into a Table2Charts corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "vizml" / "data" / "raw_data_all.csv",
        help="Raw VizML TSV/CSV file with fid, chart_data, layout and table_data columns.",
    )
    parser.add_argument(
        "--vizml-data-dir",
        type=Path,
        default=REPO_ROOT / "vizml" / "data",
        help="Directory where VizML-compatible intermediate TSV files are written.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Data" / "PlotlyTable2Charts",
        help="Output corpus root in Table2Charts layout.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=1000,
        help="Maximum table rows used for feature extraction. 0 keeps all rows.",
    )
    parser.add_argument(
        "--signature-sample-rows",
        type=int,
        default=0,
        help=(
            "Maximum values per column used when building VizML-style deduplication "
            "signatures. 0 uses complete columns."
        ),
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Optional input row cap for smoke tests. 0 means process all rows.",
    )
    parser.add_argument(
        "--max-output-tables",
        type=int,
        default=0,
        help="Optional cap on converted deduplicated tables. 0 means no cap.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N input rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output corpus directory.",
    )
    parser.add_argument(
        "--overwrite-vizml-files",
        action="store_true",
        help="Rewrite existing VizML intermediate TSV files.",
    )
    parser.add_argument(
        "--reuse-clean-files",
        action="store_true",
        help=(
            "Reuse an existing plot_data_with_all_fields_and_header.tsv for "
            "deduplication instead of rewriting VizML clean/incomplete TSV files."
        ),
    )
    return parser.parse_args()


def set_large_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8-sig")


def load_payload(value: str) -> Any:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def iter_tsv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        if "\t" not in sample and "," in sample:
            delimiter = ","
        else:
            delimiter = "\t"
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = [header for header in TSV_HEADERS if header not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            yield {header: row.get(header, "") for header in TSV_HEADERS}


def writer_for(path: Path) -> Tuple[Any, csv.writer]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(TSV_HEADERS)
    return handle, writer


def write_row(writer: csv.writer, row: Dict[str, Any]) -> None:
    writer.writerow([row.get(header, "") for header in TSV_HEADERS])


def normalized_row(fid: str, chart_data: Any, layout: Any, table_data: Any) -> Dict[str, str]:
    return {
        "fid": fid,
        "chart_data": stable_json(chart_data),
        "layout": stable_json(layout if layout is not None else {}),
        "table_data": stable_json(table_data),
    }


def ensure_plot_data_link(input_path: Path, vizml_data_dir: Path) -> Path:
    plot_data = vizml_data_dir / "plot_data.tsv"
    if plot_data.exists() or plot_data.is_symlink():
        return plot_data
    target = os.path.relpath(input_path.resolve(), plot_data.parent.resolve())
    plot_data.symlink_to(target)
    return plot_data


def read_preserved_fids() -> set[str]:
    csv_path = REPO_ROOT / "vizml" / "experiment_data" / "ground_truth_fids_99.csv"
    if not csv_path.exists():
        return set()
    preserved: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fid = row.get("fid")
            if fid:
                preserved.add(fid)
    return preserved


def first_table(table_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not isinstance(table_data, dict) or not table_data:
        return None, None
    key = next(iter(table_data.keys()))
    table = table_data.get(key)
    if not isinstance(table, dict) or not isinstance(table.get("cols"), dict):
        return None, None
    return key, table


def sorted_columns(table_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    _, table = first_table(table_data)
    if table is None:
        return []
    cols = table.get("cols", {})
    return sorted(
        [(str(name), col) for name, col in cols.items() if isinstance(col, dict)],
        key=lambda item: int(item[1].get("order", 0) or 0),
    )


def is_missing(value: Any) -> bool:
    return HELPERS.is_missing(value)


def to_float(value: Any) -> Optional[float]:
    return HELPERS.to_float(value)


def infer_general_type(values: Sequence[Any]) -> str:
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        return "c"
    numeric_ratio = sum(to_float(v) is not None for v in non_missing) / len(non_missing)
    if numeric_ratio >= 0.85:
        return "q"
    date_ratio = sum(HELPERS.looks_like_date(v) for v in non_missing) / len(non_missing)
    if date_ratio >= 0.85:
        return "t"
    return "c"


def field_characteristic(values: Sequence[Any], general_type: str) -> Any:
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        return None
    if general_type == "q":
        nums = [to_float(v) for v in non_missing]
        nums = [v for v in nums if v is not None]
        return round(sum(nums) / len(nums), 8) if nums else None
    if general_type == "t":
        return max(str(v) for v in non_missing)
    counts = Counter(str(v) for v in non_missing)
    return counts.most_common(1)[0][0] if counts else None


def table_signature(table_data: Dict[str, Any], signature_sample_rows: int = 0) -> Optional[str]:
    columns = sorted_columns(table_data)
    if not columns:
        return None
    parts: List[str] = [str(len(columns))]
    for _, col in columns:
        values = col.get("data", [])
        if not isinstance(values, list):
            values = []
        sampled_values = values[:signature_sample_rows] if signature_sample_rows > 0 else values
        general_type = infer_general_type(sampled_values)
        parts.extend(
            [
                str(len(values)),
                general_type,
                str(field_characteristic(sampled_values, general_type)),
            ]
        )
    return "\x1f".join(parts)


def safe_schema_id(fid: str) -> str:
    text = fid.replace(":", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "plotly_schema"


def raw_type_for_field(name: str, values: Sequence[Any]) -> int:
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        return RAW_TYPE_STRING
    if HELPERS.is_year_like(name, values):
        return RAW_TYPE_YEAR
    numeric_ratio = sum(to_float(v) is not None for v in non_missing) / len(non_missing)
    if numeric_ratio >= 0.85:
        return RAW_TYPE_DECIMAL
    date_ratio = sum(HELPERS.looks_like_date(v) for v in non_missing) / len(non_missing)
    if date_ratio >= 0.85:
        return RAW_TYPE_DATETIME
    return RAW_TYPE_STRING


def feature_safe_values(values: Sequence[Any], raw_type: int) -> List[Any]:
    if raw_type not in {RAW_TYPE_DECIMAL, RAW_TYPE_YEAR}:
        return list(values)
    safe: List[Any] = []
    for value in values:
        parsed = to_float(value)
        if parsed is None:
            safe.append(value)
            continue
        # Some public Plotly tables contain extremely large generated values.
        # They are valid as data, but higher moments can overflow double
        # precision during NN feature preparation. Clipping only affects the
        # feature vector, not chart labels or field identity.
        if parsed > 1e50:
            parsed = 1e50
        elif parsed < -1e50:
            parsed = -1e50
        safe.append(parsed)
    return safe


def build_table_rows(columns: List[Tuple[str, Dict[str, Any]]], max_rows: int) -> Tuple[List[str], List[List[Any]]]:
    fields = [name for name, _ in columns]
    n_rows = max((len(col.get("data", [])) for _, col in columns), default=0)
    if max_rows > 0:
        n_rows = min(n_rows, max_rows)
    rows: List[List[Any]] = []
    for row_idx in range(n_rows):
        row: List[Any] = []
        for _, col in columns:
            values = col.get("data", [])
            row.append(values[row_idx] if isinstance(values, list) and row_idx < len(values) else "")
        rows.append(row)
    return fields, rows


def make_df_and_embeddings(schema_id: str, columns: List[Tuple[str, Dict[str, Any]]], c_uids: List[str], max_rows: int) -> Tuple[Dict[str, Any], List[Any], Dict[str, int], Dict[str, int], Dict[int, List[Any]]]:
    field_names, rows = build_table_rows(columns, max_rows)
    n_rows = len(rows)
    field_types: Dict[str, int] = {}
    fields_json = []
    values_by_index: Dict[int, List[Any]] = {}
    uid_to_index: Dict[str, int] = {}

    for idx, (field_name, col) in enumerate(columns):
        values = [row[idx] for row in rows]
        values_by_index[idx] = values
        raw_type = raw_type_for_field(field_name, values)
        field_types[field_name] = raw_type
        uid = str(col.get("uid", ""))
        if uid:
            uid_to_index[uid] = idx
        fields_json.append(HELPERS.make_field(field_name, idx, feature_safe_values(values, raw_type), raw_type, n_rows))

    df_json = {
        "tUid": f"{schema_id}.t0",
        "pUids": [],
        "cUids": c_uids,
        "isExternal": False,
        "nColumns": len(field_names),
        "nRows": n_rows,
        "fields": fields_json,
    }
    embeddings = [HELPERS.make_embedding(name) for name in field_names]
    field_index = {name: idx for idx, name in enumerate(field_names)}
    return df_json, embeddings, uid_to_index, field_index, values_by_index


def plotly_trace_type(trace: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    trace_type = str(trace.get("type") or "").lower()
    mode = str(trace.get("mode") or "").lower()
    has_line = "line" in mode
    if trace.get("line") and isinstance(trace.get("line"), dict):
        has_line = True
    marker = trace.get("marker")
    if isinstance(marker, dict) and isinstance(marker.get("line"), dict):
        if marker["line"].get("color") != "transparent":
            has_line = True

    if "bar" in trace_type:
        return "barChart", has_line
    if "pie" in trace_type:
        return "pieChart", has_line
    if "line" in trace_type:
        return "lineChart", True
    if "scatter" in trace_type and "scatter3d" not in trace_type:
        return ("lineChart" if has_line else "scatterChart"), has_line
    if not trace_type and has_line:
        return "lineChart", True
    return None, has_line


def source_to_index(src: Any, uid_to_index: Dict[str, int]) -> Optional[int]:
    if not isinstance(src, str) or not src:
        return None
    uid = src.split(":")[-1]
    return uid_to_index.get(uid)


def numeric_record(values: Sequence[Any]) -> List[float]:
    nums: List[float] = []
    for value in values:
        parsed = to_float(value)
        if parsed is None:
            raise ValueError("non numeric")
        nums.append(parsed)
    return nums


def monotonic(values: Sequence[Any]) -> bool:
    try:
        nums = numeric_record([v for v in values if not is_missing(v)])
    except ValueError:
        return False
    if len(nums) < 2:
        return True
    return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) or all(
        nums[i] >= nums[i + 1] for i in range(len(nums) - 1)
    )


def normalize_ana_type(ana_type: str, x_idx: Optional[int], field_types_by_index: Dict[int, int], values_by_index: Dict[int, List[Any]], draws_line: bool) -> Optional[str]:
    if ana_type == "scatterChart":
        if x_idx is None:
            return None
        x_type = field_types_by_index.get(x_idx)
        if x_type == RAW_TYPE_STRING:
            return None
        return "scatterChart"
    if ana_type == "lineChart":
        if not draws_line:
            return None
        return "lineChart"
    return ana_type


def chart_key(trace: Dict[str, Any], ana_type: str) -> Tuple[Any, ...]:
    return (
        trace.get("xsrc") or trace.get("labelsrc"),
        trace.get("xaxis"),
        trace.get("yaxis"),
        ana_type,
    )


def parse_plotly_charts(
    schema_id: str,
    chart_data: List[Dict[str, Any]],
    layout: Dict[str, Any],
    uid_to_index: Dict[str, int],
    df_json: Dict[str, Any],
    values_by_index: Dict[int, List[Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    stats = Counter()
    field_types_by_index = {field["index"]: field["type"] for field in df_json["fields"]}
    grouped: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()

    barmode = str((layout or {}).get("barmode") or "").lower()
    grouping = "stacked" if barmode in {"stack", "relative"} else "clustered"

    for trace in chart_data:
        if not isinstance(trace, dict):
            stats["skip:bad_trace"] += 1
            continue
        trace = dict(trace)
        if trace.get("orientation") == "h":
            trace["xsrc"], trace["ysrc"] = trace.get("ysrc"), trace.get("xsrc")
            trace["xaxis"], trace["yaxis"] = trace.get("yaxis"), trace.get("xaxis")
        if trace.get("valuesrc") is not None:
            trace["ysrc"] = trace.get("valuesrc")
        if trace.get("labelsrc") is not None and not trace.get("xsrc"):
            trace["xsrc"] = trace.get("labelsrc")

        ana_type, draws_line = plotly_trace_type(trace)
        if ana_type not in SUPPORTED_ANA:
            stats["skip:unsupported_type"] += 1
            continue

        x_idx = source_to_index(trace.get("xsrc"), uid_to_index)
        y_idx = source_to_index(trace.get("ysrc"), uid_to_index)
        if y_idx is None:
            stats["skip:missing_y"] += 1
            continue
        if field_types_by_index.get(y_idx) == RAW_TYPE_STRING:
            stats["skip:string_y"] += 1
            continue
        if ana_type != "pieChart" and x_idx is None:
            stats["skip:missing_x"] += 1
            continue

        normalized_ana = normalize_ana_type(ana_type, x_idx, field_types_by_index, values_by_index, draws_line)
        if normalized_ana is None:
            stats[f"skip:invalid_{ana_type}"] += 1
            continue

        # Scatter traces with explicit lines over a monotonic x-axis are better
        # represented as line charts for this action grammar.
        if ana_type == "scatterChart" and draws_line and x_idx is not None and monotonic(values_by_index.get(x_idx, [])):
            normalized_ana = "lineChart"

        key = chart_key(trace, normalized_ana)
        if key not in grouped:
            grouped[key] = {
                "anaType": normalized_ana,
                "xFields": [] if x_idx is None else [{"index": x_idx}],
                "yFields": [],
                "valueDrawsLine": [],
                "grouping": grouping,
            }
        item = grouped[key]
        if {"index": y_idx} not in item["yFields"]:
            item["yFields"].append({"index": y_idx})
            item["valueDrawsLine"].append(bool(draws_line))

    chart_files: List[Dict[str, Any]] = []
    sample_items: List[Dict[str, Any]] = []
    for chart_idx, chart in enumerate(grouped.values()):
        if not chart["yFields"]:
            continue
        out_chart = {
            "xFields": chart["xFields"],
            "yFields": chart["yFields"],
        }
        if chart["anaType"] == "barChart":
            out_chart["grouping"] = chart["grouping"]
        if chart["anaType"] in {"lineChart", "scatterChart"}:
            out_chart["valueDrawsLine"] = chart["valueDrawsLine"]
        chart_files.append(out_chart)
        sample_items.append(
            {
                "anaType": chart["anaType"],
                "nVals": len(chart["yFields"]),
                "index": str(chart_idx),
            }
        )
        stats[f"ana:{chart['anaType']}"] += 1

    return chart_files, sample_items, stats


def prepare_output_dirs(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    for subdir in ("data", "sample-new", "index", "embeddings/fasttext", "raw/data_origin"):
        (output / subdir).mkdir(parents=True, exist_ok=True)


def remove_if_allowed(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite-vizml-files to rewrite it")
    path.unlink()


def scan_clean_and_deduplicate(args: argparse.Namespace) -> Tuple[set[str], Dict[str, Any]]:
    args.vizml_data_dir.mkdir(parents=True, exist_ok=True)
    ensure_plot_data_link(args.input, args.vizml_data_dir)

    clean_path = args.vizml_data_dir / "plot_data_with_all_fields_and_header.tsv"
    incomplete_path = args.vizml_data_dir / "plot_data_without_all_fields_and_header.tsv"
    if args.reuse_clean_files:
        if not clean_path.exists():
            raise SystemExit(f"{clean_path} does not exist; cannot --reuse-clean-files")
        clean_handle = clean_writer = None
        incomplete_handle = incomplete_writer = None
        scan_source = clean_path
    else:
        remove_if_allowed(clean_path, args.overwrite_vizml_files)
        remove_if_allowed(incomplete_path, args.overwrite_vizml_files)
        clean_handle, clean_writer = writer_for(clean_path)
        incomplete_handle, incomplete_writer = writer_for(incomplete_path)
        scan_source = args.input

    preserved_fids = read_preserved_fids()
    signature_to_fid: Dict[str, str] = {}
    stats = Counter()
    started = time.time()

    try:
        for row_num, row in enumerate(iter_tsv_rows(scan_source), start=1):
            if args.limit_rows and row_num > args.limit_rows:
                break
            stats["rows_seen"] += 1
            try:
                chart_data = load_payload(row["chart_data"])
                layout = load_payload(row["layout"]) or {}
                table_data = load_payload(row["table_data"])
            except Exception:
                stats["parse_errors"] += 1
                continue

            fid = row["fid"]
            if not chart_data or not table_data:
                stats["incomplete"] += 1
                if incomplete_writer is not None:
                    write_row(incomplete_writer, normalized_row(fid, chart_data or [], layout, table_data or {}))
                continue

            if clean_writer is not None:
                write_row(clean_writer, normalized_row(fid, chart_data, layout, table_data))
            stats["clean"] += 1

            signature = table_signature(table_data, args.signature_sample_rows)
            if signature is None:
                stats["signature_errors"] += 1
                continue
            old_fid = signature_to_fid.get(signature)
            if old_fid is None or fid in preserved_fids:
                signature_to_fid[signature] = fid
                if old_fid is None:
                    stats["unique_signatures"] += 1
                elif old_fid != fid:
                    stats["preserved_replacements"] += 1
            else:
                stats["duplicates"] += 1

            if args.progress_every and row_num % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"[scan] rows={row_num:,} clean={stats['clean']:,} "
                    f"unique={len(signature_to_fid):,} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if clean_handle is not None:
            clean_handle.close()
        if incomplete_handle is not None:
            incomplete_handle.close()

    keep_fids = set(signature_to_fid.values())
    return keep_fids, {
        "stats": dict(stats),
        "clean_path": rel(clean_path),
        "incomplete_path": rel(incomplete_path),
        "kept_fids": len(keep_fids),
        "reused_clean_files": args.reuse_clean_files,
    }


def convert_kept_rows(args: argparse.Namespace, keep_fids: set[str]) -> Dict[str, Any]:
    clean_path = args.vizml_data_dir / "plot_data_with_all_fields_and_header.tsv"
    dedup_path = args.vizml_data_dir / "plot_data_with_all_fields_and_header_deduplicated_one_per_user.tsv"
    remove_if_allowed(dedup_path, args.overwrite_vizml_files)
    dedup_handle, dedup_writer = writer_for(dedup_path)

    prepare_output_dirs(args.output, args.overwrite)
    raw_tsv_path = args.output / "plotly_data_dedup.tsv"
    raw_tsv_handle, raw_tsv_writer = writer_for(raw_tsv_path)

    schema_ids: List[str] = []
    fid_map: Dict[str, str] = {}
    stats = Counter()
    started = time.time()

    try:
        for row_num, row in enumerate(iter_tsv_rows(clean_path), start=1):
            fid = row["fid"]
            if fid not in keep_fids:
                continue
            if args.max_output_tables and stats["converted_tables"] >= args.max_output_tables:
                break

            stats["dedup_rows"] += 1
            write_row(dedup_writer, row)

            try:
                chart_data = load_payload(row["chart_data"])
                layout = load_payload(row["layout"]) or {}
                table_data = load_payload(row["table_data"])
            except Exception:
                stats["convert_parse_errors"] += 1
                continue
            if not isinstance(chart_data, list) or not isinstance(table_data, dict):
                stats["skip:bad_payload"] += 1
                continue
            table_key, table = first_table(table_data)
            if table is None:
                stats["skip:bad_table"] += 1
                continue
            columns = sorted_columns(table_data)
            if not columns:
                stats["skip:no_columns"] += 1
                continue

            schema_id = safe_schema_id(fid)
            # Avoid rare collisions after sanitizing.
            base_schema_id = schema_id
            suffix = 1
            while schema_id in fid_map:
                suffix += 1
                schema_id = f"{base_schema_id}_{suffix}"
            fid_map[schema_id] = fid

            df_json, embeddings, uid_to_index, _, values_by_index = make_df_and_embeddings(
                schema_id=schema_id,
                columns=columns,
                c_uids=[],
                max_rows=args.max_source_rows,
            )
            chart_files, sample_items, chart_stats = parse_plotly_charts(
                schema_id=schema_id,
                chart_data=chart_data,
                layout=layout,
                uid_to_index=uid_to_index,
                df_json=df_json,
                values_by_index=values_by_index,
            )
            stats.update(chart_stats)
            if not sample_items:
                stats["skip:no_supported_charts"] += 1
                continue

            c_uids = [f"{schema_id}.t0.c{i}" for i in range(len(chart_files))]
            df_json["cUids"] = c_uids
            dump_json(df_json, args.output / "data" / f"{schema_id}.t0.DF.json")
            dump_json(embeddings, args.output / "embeddings" / "fasttext" / f"{schema_id}.EMB.json")
            for idx, chart_json in enumerate(chart_files):
                dump_json(chart_json, args.output / "data" / f"{schema_id}.t0.c{idx}.json")
            sample_json = {
                "sID": schema_id,
                "lang": "en",
                "nColumns": len(columns),
                "tableAnalysisPairs": {"0": sample_items},
            }
            dump_json(sample_json, args.output / "sample-new" / f"{schema_id}.sample.json")

            # Compatibility artifacts for the documented ChartSplit input shape.
            dump_json(chart_data, args.output / "raw" / "data_origin" / f"{schema_id}_chartdata.json")
            dump_json(table_data, args.output / "raw" / "data_origin" / f"{schema_id}_tabledata.json")
            raw_tsv_writer.writerow(
                [
                    schema_id,
                    row["chart_data"],
                    row["layout"],
                    row["table_data"],
                ]
            )

            schema_ids.append(schema_id)
            stats["converted_tables"] += 1
            stats["converted_charts"] += len(sample_items)
            stats["fields"] += len(columns)
            stats["rows_for_features"] += df_json["nRows"]

            if args.progress_every and stats["dedup_rows"] % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"[convert] dedup={stats['dedup_rows']:,} converted={stats['converted_tables']:,} "
                    f"charts={stats['converted_charts']:,} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        dedup_handle.close()
        raw_tsv_handle.close()

    dump_json(schema_ids, args.output / "index" / "schema_ids.json")
    dump_json(fid_map, args.output / "raw" / "fid_map.json")

    return {
        "stats": dict(stats),
        "dedup_path": rel(dedup_path),
        "output": rel(args.output),
        "schema_ids": len(schema_ids),
        "raw_chartsplit_compatible_tsv": rel(raw_tsv_path),
    }


def main() -> int:
    set_large_csv_field_limit()
    args = parse_args()
    args.input = args.input.resolve()
    args.vizml_data_dir = args.vizml_data_dir.resolve()
    args.output = args.output.resolve()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Input: {rel(args.input)}")
    keep_fids, scan_report = scan_clean_and_deduplicate(args)
    print(f"Kept deduplicated fids: {len(keep_fids):,}")
    convert_report = convert_kept_rows(args, keep_fids)

    report = {
        "input": rel(args.input),
        "max_source_rows": args.max_source_rows,
        "signature_sample_rows": args.signature_sample_rows,
        "limit_rows": args.limit_rows,
        "max_output_tables": args.max_output_tables,
        "scan": scan_report,
        "convert": convert_report,
    }
    dump_json(report, args.output / "conversion_report.json")

    totals = convert_report["stats"]
    print(f"Converted tables: {totals.get('converted_tables', 0):,}")
    print(f"Converted charts: {totals.get('converted_charts', 0):,}")
    for key, value in sorted(totals.items()):
        if key.startswith("ana:"):
            print(f"{key}: {value:,}")
    print(f"Report: {rel(args.output / 'conversion_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
