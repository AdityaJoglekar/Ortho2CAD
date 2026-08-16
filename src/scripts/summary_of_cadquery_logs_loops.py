"""Summarize CadQuery loop log IoUs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _coerce_iou(iou_obj: Any) -> float:
	"""Extract IoU as float; null/missing/invalid becomes 0.0; clip to [0, 1]."""

	if isinstance(iou_obj, dict):
		value = iou_obj.get("iou")
	else:
		value = iou_obj

	if value is None:
		v = 0.0
	else:
		try:
			v = float(value)
		except (TypeError, ValueError):
			v = 0.0

	if not math.isfinite(v):
		v = 0.0

	if v < 0.0:
		return 0.0
	if v > 1.0:
		return 1.0
	return v


def _is_iou_error(iou_obj: Any) -> bool:
	"""Count missing, failed, or invalid IoU records as errors for that pass."""

	if iou_obj is None:
		return True
	if isinstance(iou_obj, dict):
		if iou_obj.get("ok") is False:
			return True
		if iou_obj.get("error"):
			return True
		value = iou_obj.get("iou")
	else:
		value = iou_obj
	if value is None:
		return True
	try:
		return not math.isfinite(float(value))
	except (TypeError, ValueError):
		return True


def _read_ious_dict(data: dict[str, Any]) -> dict[str, Any]:
	raw = data.get("ious")
	return raw if isinstance(raw, dict) else {}


def _stats(values: list[float]) -> dict[str, float]:
	if not values:
		return {"mean": 0.0, "min": 0.0, "max": 0.0, "std_dev": 0.0}
	mean = sum(values) / len(values)
	variance = sum((value - mean) ** 2 for value in values) / len(values)
	return {
		"mean": mean,
		"min": min(values),
		"max": max(values),
		"std_dev": math.sqrt(variance),
	}


def _loop_index(pass_name: str) -> int | None:
	m = re.fullmatch(r"loop(\d+)(?:_final)?", pass_name)
	return int(m.group(1)) if m else None


def _expected_pass_names(max_loop_idx: int) -> list[str]:
	pass_names = ["exec0", "exec0_final"]
	for loop_idx in range(1, max_loop_idx + 1):
		pass_names.extend([f"loop{loop_idx}", f"loop{loop_idx}_final"])
	return pass_names


def _load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def _save_final_iou_hist(
	values: list[float], *, bins: int, lo: float, hi: float, title: str, out_path: Path
) -> None:
	try:
		import matplotlib  # type: ignore

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt  # type: ignore
	except Exception as e:  # pragma: no cover
		raise SystemExit(
			"matplotlib is required for plotting. "
			"Run via `conda run -n pyocc python ...` or install matplotlib in your env. "
			f"Import error: {e}"
		)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 4.5), dpi=150)
	plt.hist(values, bins=bins, range=(lo, hi), edgecolor="black", linewidth=0.5)
	plt.title(title)
	plt.xlabel("IoU")
	plt.ylabel("Count")
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def summarize_logs(
	logs_dir: Path,
	*,
	pattern: str = "*.json",
	bins: int = 20,
	hist_range: tuple[float, float] = (0.0, 1.0),
	out_dir: Path | None = None,
) -> dict[str, Any]:
	log_paths = sorted(logs_dir.glob(pattern))
	if not log_paths:
		raise SystemExit(f"No logs found in {logs_dir} matching {pattern!r}")

	lo, hi = hist_range
	if hi <= lo:
		raise SystemExit(f"Invalid --hist-range: HI ({hi}) must be greater than LO ({lo})")

	log_data = [_load_json(path) for path in log_paths]
	max_loop_idx = 0
	for data in log_data:
		for pass_name in _read_ious_dict(data):
			if not isinstance(pass_name, str):
				continue
			loop_idx = _loop_index(pass_name)
			if loop_idx is not None:
				max_loop_idx = max(max_loop_idx, loop_idx)

	ordered_passes = _expected_pass_names(max_loop_idx)
	iou_values_by_pass: dict[str, list[float]] = {}
	errors_by_pass: dict[str, int] = {pass_name: 0 for pass_name in ordered_passes}
	final_iou_values: list[float] = []
	best_iou_values: list[float] = []

	for data in log_data:
		iou_map = _read_ious_dict(data)

		sample_iou_values: list[float] = []
		for pass_name in ordered_passes:
			raw_iou = iou_map.get(pass_name)
			iou_value = _coerce_iou(raw_iou)
			iou_values_by_pass.setdefault(pass_name, []).append(iou_value)
			if _is_iou_error(raw_iou):
				errors_by_pass[pass_name] += 1
			sample_iou_values.append(iou_value)

		best_iou_values.append(max(sample_iou_values) if sample_iou_values else 0.0)
		final_iou_values.append(_coerce_iou(data.get("final_iou")))

	if out_dir is None:
		out_dir = logs_dir.parent / "summary"
	out_dir.mkdir(parents=True, exist_ok=True)

	histogram_path = out_dir / "final_iou_hist.png"
	_save_final_iou_hist(
		final_iou_values,
		bins=bins,
		lo=lo,
		hi=hi,
		title=f"final_iou (n={len(log_paths)})",
		out_path=histogram_path,
	)

	summary: dict[str, Any] = {
		"ious": {
			pass_name: _stats(iou_values_by_pass[pass_name]) for pass_name in ordered_passes
		},
		"errors": errors_by_pass,
		"final_iou": _stats(final_iou_values),
		"best_iou": _stats(best_iou_values),
	}

	return summary


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Summarize CadQuery loop log IoUs")
	### Change the path to the logs directory here as per requirements
	parser.add_argument(
		"--logs-dir",
		type=Path,
		default=Path(
			"../inference/inference_results/gpt-5.5/f360rec_test_data_subset100_mdim_api_rewrite_2loops/logs"
			# "../inference/inference_results/claude-opus-4-8/f360rec_test_data_subset100_mdim_api_rewrite_2loops/logs"
		),
		help="Directory containing per-sample JSON logs",
	)
	parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern within logs-dir")
	parser.add_argument("--bins", type=int, default=20, help="Histogram bins for final_iou PNG")
	parser.add_argument(
		"--hist-range",
		nargs=2,
		type=float,
		default=(0.0, 1.0),
		metavar=("LO", "HI"),
		help="Histogram x-range (IoUs clipped to [0,1])",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Output directory for summary + plot (default: <logs-dir>/../summary)",
	)
	parser.add_argument(
		"--summary-path",
		type=Path,
		default=None,
		help="JSON summary path (default: <out-dir>/cadquery_log_summary.json)",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	hist_range = (float(args.hist_range[0]), float(args.hist_range[1]))
	out_dir: Path | None = args.out_dir

	summary = summarize_logs(
		args.logs_dir,
		pattern=args.pattern,
		bins=args.bins,
		hist_range=hist_range,
		out_dir=out_dir,
	)

	resolved_out_dir = args.out_dir or (args.logs_dir.parent / "summary")
	summary_path = args.summary_path or (resolved_out_dir / "cadquery_log_summary.json")
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
