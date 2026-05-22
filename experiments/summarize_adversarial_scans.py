"""Summarize adversarial scan CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def endpoint_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("_source", ""),
        row.get("scan_type", ""),
        row.get("dimension", ""),
        row.get("endpoint", row.get("sample", "")),
        row.get("ratio", ""),
        row.get("mean_norm", ""),
        row.get("angle", ""),
        row.get("seed", ""),
    )


def endpoint_rows(rows: list[dict[str, str]]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(endpoint_key(row), []).append(row)
    return grouped


def finite_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = [_float(row, key) for row in rows]
    return [value for value in values if math.isfinite(value)]


def suspicious_score(row: dict[str, str]) -> float:
    score = 0.0
    hessian = _float(row, "hessian_min_eigenvalue", _float(row, "refined_hessian"))
    horizontal = _float(row, "horizontal_residual", _float(row, "refined_horizontal_residual", 0.0))
    endpoint_error = _float(row, "endpoint_error", 0.0)
    precision_error = _float(row, "precision_endpoint_error", math.nan)
    lifted_error = _float(row, "lifted_endpoint_error", math.nan)
    projection_failure_class = row.get("projection_failure_class", "")
    omega_spread = _float(row, "omega_spread", _float(row, "optimizer_spread", 0.0))
    if math.isfinite(hessian):
        score += max(0.0, -math.log10(max(abs(hessian), 1e-300)))
        if hessian < 0.0:
            score += 20.0
    if math.isfinite(horizontal):
        score += max(0.0, math.log10(max(horizontal, 1e-300)) + 10.0)
    if projection_failure_class == "projection_conditioning":
        score += 2.0
    elif math.isfinite(precision_error) and math.isfinite(lifted_error) and precision_error < 1e-8 and lifted_error < 1e-8:
        score += 2.0
    elif math.isfinite(endpoint_error):
        score += max(0.0, math.log10(max(endpoint_error, 1e-300)) + 10.0)
    if math.isfinite(precision_error):
        score += max(0.0, math.log10(max(precision_error, 1e-300)) + 10.0)
    if math.isfinite(omega_spread):
        score += max(0.0, math.log10(max(omega_spread, 1e-300)) + 8.0)
    score += 10.0 * _int(row, "multiple_apparent_minima")
    score += 5.0 * _int(row, "multiple_critical_points")
    if _int(row, "success", 1) == 0:
        score += 5.0
    return score


def print_summary(rows: list[dict[str, str]]) -> None:
    grouped = endpoint_rows(rows)
    horizontal_values = finite_values(rows, "horizontal_residual") + finite_values(rows, "refined_horizontal_residual")
    endpoint_errors = finite_values(rows, "endpoint_error")
    lifted_errors = finite_values(rows, "lifted_endpoint_error") + finite_values(rows, "best_lifted_endpoint_error")
    precision_errors = finite_values(rows, "precision_endpoint_error") + finite_values(rows, "best_precision_endpoint_error")
    covariance_errors = finite_values(rows, "covariance_endpoint_error") + finite_values(rows, "best_covariance_endpoint_error")
    hessian_values = finite_values(rows, "hessian_min_eigenvalue") + finite_values(rows, "refined_hessian") + finite_values(rows, "smallest_critical_hessian")
    spreads = finite_values(rows, "omega_spread") + finite_values(rows, "optimizer_spread")
    multiple_minima = sum(
        1
        for group in grouped.values()
        if any(max(_int(row, "multiple_apparent_minima"), _int(row, "multiple_minima")) for row in group)
    )
    multiple_critical = sum(1 for group in grouped.values() if any(_int(row, "multiple_critical_points") for row in group))
    print(f"total rows: {len(rows)}")
    print(f"total endpoints scanned: {len(grouped)}")
    print(f"endpoints with multiple apparent minima: {multiple_minima}")
    print(f"endpoints with multiple critical points: {multiple_critical}")
    print(f"worst horizontal residual: {max(horizontal_values) if horizontal_values else math.nan:.12g}")
    print(f"worst endpoint error: {max(endpoint_errors) if endpoint_errors else math.nan:.12g}")
    print(f"worst lifted endpoint error: {max(lifted_errors) if lifted_errors else math.nan:.12g}")
    print(f"worst precision endpoint error: {max(precision_errors) if precision_errors else math.nan:.12g}")
    print(f"worst covariance endpoint error: {max(covariance_errors) if covariance_errors else math.nan:.12g}")
    print(f"smallest Hessian eigenvalue: {min(hessian_values) if hessian_values else math.nan:.12g}")
    print(f"largest optimizer spread: {max(spreads) if spreads else math.nan:.12g}")
    print("")
    print("top 10 suspicious endpoints:")
    ranked = sorted(rows, key=suspicious_score, reverse=True)[:10]
    for index, row in enumerate(ranked, start=1):
        print(
            f"{index:2d}. score={suspicious_score(row):.3g} "
            f"source={row.get('_source')} scan={row.get('scan_type')} d={row.get('dimension')} "
            f"endpoint={row.get('endpoint', row.get('sample', ''))} start={row.get('start', '')} "
            f"cond={row.get('condition', '')} mu_norm={row.get('mean_norm', '')} "
            f"energy={row.get('energy', row.get('refined_energy', ''))} "
            f"horiz={row.get('horizontal_residual', row.get('refined_horizontal_residual', ''))} "
            f"prec_err={row.get('precision_endpoint_error', row.get('best_precision_endpoint_error', ''))} "
            f"cov_err={row.get('covariance_endpoint_error', row.get('best_covariance_endpoint_error', row.get('endpoint_error', '')))} "
            f"class={row.get('projection_failure_class', row.get('best_projection_failure_class', ''))} "
            f"hess={row.get('hessian_min_eigenvalue', row.get('refined_hessian', ''))} "
            f"spread={row.get('omega_spread', row.get('optimizer_spread', ''))} "
            f"status={row.get('solver_status', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+")
    args = parser.parse_args()
    print_summary(read_rows(args.csv))


if __name__ == "__main__":
    main()
