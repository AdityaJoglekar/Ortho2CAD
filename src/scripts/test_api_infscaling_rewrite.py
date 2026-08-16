#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import textwrap
import uuid
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

try:
	from anthropic import Anthropic  # type: ignore[import-not-found]
except ImportError:
	Anthropic = None


SCRIPTS_DIR = "./scripts"

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
	rows: List[Dict[str, Any]] = []
	with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if not isinstance(obj, dict):
				raise ValueError(f"Expected JSON object on line {line_no} of {path}, got {type(obj)}")
			rows.append(obj)
	return rows


def _encode_image_as_data_url(image_path: str) -> str:
	# Encode local images once, then adapt the same bytes to each provider's image block shape.
	ext = os.path.splitext(image_path)[1].lower()
	mime = "image/png" if ext == ".png" else "image/jpeg" if ext in {".jpg", ".jpeg"} else "application/octet-stream"
	with open(image_path, "rb") as f:
		b64 = base64.b64encode(f.read()).decode("ascii")
	return f"data:{mime};base64,{b64}"


def _strip_code_fences(text: str) -> str:
	t = text.strip()
	if t.startswith("```"):
		# Remove first fence line
		first_newline = t.find("\n")
		if first_newline != -1:
			t = t[first_newline + 1 :]
		# Remove trailing fence
		if t.rstrip().endswith("```"):
			t = t.rstrip()
			t = t[: -3]
	return t.strip()


def _tail(text: str, max_chars: int = 8000) -> str:
	if len(text) <= max_chars:
		return text
	return text[-max_chars:]

def _infer_api_provider(model: str, requested_provider: str = "auto") -> str:
	if requested_provider != "auto":
		return requested_provider
	model_l = model.lower()
	if model_l.startswith("claude") or model_l.startswith("anthropic/claude"):
		print("Using Anthropic")
		return "anthropic"
	print("Using OpenAI")
	return "openai"


def _get_api_client(provider: str) -> Dict[str, Any]:
	if provider == "openai":
		return {"provider": provider, "client": OpenAI()}
	if provider == "anthropic":
		if Anthropic is None:
			raise ImportError("The 'anthropic' package is required for --api-provider anthropic. Install it with `pip install anthropic`.")
		return {"provider": provider, "client": Anthropic()}
	raise ValueError(f"Unsupported API provider: {provider}")


def _run_conda_python(
	*,
	conda_env: str,
	args: List[str],
	timeout_s: float,
	cwd: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
	cmd = ["conda", "run", "-n", conda_env, "python", *args]
	return subprocess.run(
		cmd,
		cwd=cwd,
		text=True,
		capture_output=True,
		timeout=timeout_s,
	)


def _write_text(path: str, content: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		f.write(content)


def _cadquery_runner_code(*, cadquery_code: str, step_path: str, tmp_step_path: str) -> str:
	# We inject an export line and a minimal validation to ensure the model assigns the final solid to variable 'solid'.
	model_code = _strip_code_fences(cadquery_code).strip()
	return (
		"import os\n"
		"import cadquery as cq\n"
		"\n"
		"# --- model code (begin) ---\n"
		f"{model_code}\n"
		"# --- model code (end) ---\n"
		"\n"
		"if 'solid' not in locals():\n"
		"    raise NameError(\"CadQuery code must assign the final solid to variable 'solid'.\")\n"
		f"cq.exporters.export(solid, r\"{tmp_step_path}\", exportType=\"STEP\")\n"
		f"os.replace(r\"{tmp_step_path}\", r\"{step_path}\")\n"
		"print('EXPORT_OK')\n"
	)


def execute_cadquery_to_step(
	*,
	conda_env: str,
	qid: str,
	pass_name: str,
	cadquery_code: str,
	step_path: str,
	work_dir: str,
	timeout_s: float,
) -> Dict[str, Any]:
	"""Execute CadQuery code in a conda env and export STEP.

	Returns a dict with: ok(bool), stdout, stderr, error(str|None)
	"""
	os.makedirs(work_dir, exist_ok=True)
	# Resolve paths for filesystem operations, but write relative paths into the generated runner.
	work_dir_abs = os.path.abspath(os.path.expanduser(work_dir))
	step_path_abs = os.path.abspath(os.path.expanduser(step_path))
	os.makedirs(os.path.dirname(step_path_abs), exist_ok=True)
	tmp_step_path_abs = step_path_abs + f".tmp_{uuid.uuid4().hex}"
	runner = _cadquery_runner_code(
		cadquery_code=cadquery_code,
		step_path=os.path.relpath(step_path_abs, work_dir_abs),
		tmp_step_path=os.path.relpath(tmp_step_path_abs, work_dir_abs),
	)
	script_name = f"{qid}.{pass_name}.cadquery_export.py"
	script_path = os.path.join(work_dir, script_name)
	_write_text(script_path, runner)
	# Export to a temp file and atomically replace destination on success.

	try:
		# With cwd=work_dir, pass only the filename.
		proc = _run_conda_python(conda_env=conda_env, args=[script_name], timeout_s=timeout_s, cwd=work_dir)
	except subprocess.TimeoutExpired as e:
		return {
			"ok": False,
			"stdout": _tail(getattr(e, "stdout", "") or ""),
			"stderr": _tail(getattr(e, "stderr", "") or ""),
			"error": f"Timed out after {timeout_s}s",
		}

	stdout = proc.stdout or ""
	stderr = proc.stderr or ""
	ok = proc.returncode == 0 and os.path.exists(step_path_abs)
	if ok:
		return {"ok": True, "stdout": _tail(stdout), "stderr": _tail(stderr), "error": None, "script_path": script_path}

	err_msg = f"Return code: {proc.returncode}\nSTDOUT:\n{_tail(stdout)}\n\nSTDERR:\n{_tail(stderr)}"
	if proc.returncode == 0 and not os.path.exists(step_path_abs):
		err_msg = "STEP export did not produce a file.\n" + err_msg
	return {"ok": False, "stdout": _tail(stdout), "stderr": _tail(stderr), "error": err_msg, "script_path": script_path}


def _write_json(path: str, obj: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def _relative_path_for_output(path: str, *, base_dir: Optional[str] = None) -> str:
	base = os.path.abspath(base_dir or os.getcwd())
	return os.path.relpath(os.path.abspath(os.path.expanduser(path)), base)


def _sanitize_text_paths_for_output(text: str, *, base_dir: Optional[str] = None) -> str:
	def repl(match) -> str:
		raw_path = match.group(0)
		trailing = ""
		while raw_path and raw_path[-1] in ".,:;)]}":
			trailing = raw_path[-1] + trailing
			raw_path = raw_path[:-1]
		if not raw_path:
			return match.group(0)
		return _relative_path_for_output(raw_path, base_dir=base_dir) + trailing

	return re.sub(r"(?<![:/])/[^\s'\"<>]+", repl, text)


def _path_key_for_output(key: Optional[str]) -> bool:
	if key is None:
		return False
	return (
		key in {"image", "gt_step", "ortho_png", "ortho_svg", "source_ortho_png"}
		or key.endswith("_path")
		or key.endswith("_dir")
		or key.endswith("_png")
		or key.endswith("_svg")
	)


def _make_paths_relative_for_output(obj: Any, *, key: Optional[str] = None, base_dir: Optional[str] = None) -> Any:
	if isinstance(obj, dict):
		return {
			k: _make_paths_relative_for_output(v, key=str(k), base_dir=base_dir)
			for k, v in obj.items()
		}
	if isinstance(obj, list):
		return [_make_paths_relative_for_output(v, key=key, base_dir=base_dir) for v in obj]
	if isinstance(obj, str):
		if _path_key_for_output(key) and os.path.isabs(os.path.expanduser(obj)):
			return _relative_path_for_output(obj, base_dir=base_dir)
		if key in {"error", "stdout", "stderr", "step_copy_error", "gt_step_resolve_error", "ortho_error", "ortho_final_error"}:
			return _sanitize_text_paths_for_output(obj, base_dir=base_dir)
	return obj


def render_step_to_ortho_png(
	*,
	conda_env: str,
	step_path: str,
	svg_path: str,
	png_path: str,
	timeout_s: float,
	occ_timeout_s: int,
) -> Dict[str, Any]:
	"""Render STEP to an orthographic SVG (pythonocc), then convert to PNG (cairosvg)."""
	os.makedirs(os.path.dirname(svg_path), exist_ok=True)
	os.makedirs(os.path.dirname(png_path), exist_ok=True)

	# 1) STEP -> SVG
	occ_snippet = textwrap.dedent(
		f"""
		import os, sys
		sys.path.insert(0, r"{SCRIPTS_DIR}")
		from pythonocc_for_step_to_ortho import process_single_step_file
		ok = process_single_step_file(r"{step_path}", r"{svg_path}", verbose=False, timeout_seconds={occ_timeout_s})
		sys.exit(0 if ok else 2)
		"""
	).lstrip()
	try:
		proc1 = _run_conda_python(conda_env=conda_env, args=["-c", occ_snippet], timeout_s=timeout_s)
	except subprocess.TimeoutExpired:
		return {"ok": False, "error": f"Ortho rendering timed out after {timeout_s}s"}
	if proc1.returncode != 0 or not os.path.exists(svg_path):
		err = f"STEP->SVG failed (rc={proc1.returncode}).\nSTDOUT:\n{_tail(proc1.stdout or '')}\n\nSTDERR:\n{_tail(proc1.stderr or '')}"
		return {"ok": False, "error": err}

	# 2) SVG -> PNG
	png_snippet = textwrap.dedent(
		f"""
		import cairosvg
		cairosvg.svg2png(url=r"{svg_path}", write_to=r"{png_path}")
		print("PNG_OK")
		"""
	).lstrip()
	try:
		proc2 = _run_conda_python(conda_env=conda_env, args=["-c", png_snippet], timeout_s=timeout_s)
	except subprocess.TimeoutExpired:
		return {"ok": False, "error": f"SVG->PNG conversion timed out after {timeout_s}s"}
	if proc2.returncode != 0 or not os.path.exists(png_path):
		err = f"SVG->PNG failed (rc={proc2.returncode}).\nSTDOUT:\n{_tail(proc2.stdout or '')}\n\nSTDERR:\n{_tail(proc2.stderr or '')}"
		return {"ok": False, "error": err}

	return {"ok": True, "error": None}


def compute_step_iou(
	*,
	conda_env: str,
	pred_step_path: str,
	gt_step_path: str,
	timeout_s: float,
) -> Dict[str, Any]:
	"""Compute IoU using cq_align_shapes from compute_iou.py."""
	snippet = textwrap.dedent(
		f"""
		import re, sys
		sys.path.insert(0, r"{SCRIPTS_DIR}")
		import cadquery as cq
		from compute_iou import cq_align_shapes
		gt = cq.importers.importStep(r"{gt_step_path}")
		pred = cq.importers.importStep(r"{pred_step_path}")
		_, iou, _, _ = cq_align_shapes(pred, gt)
		print("IOU_RESULT", float(iou))
		"""
	).lstrip()
	try:
		proc = _run_conda_python(conda_env=conda_env, args=["-c", snippet], timeout_s=timeout_s)
	except subprocess.TimeoutExpired:
		return {"ok": False, "iou": None, "error": f"IoU computation timed out after {timeout_s}s"}
	if proc.returncode != 0:
		return {
			"ok": False,
			"iou": None,
			"error": f"IoU computation failed (rc={proc.returncode}).\nSTDOUT:\n{_tail(proc.stdout or '')}\n\nSTDERR:\n{_tail(proc.stderr or '')}",
		}
	# Parse IOU_RESULT
	m = re.search(r"IOU_RESULT\s+([0-9]*\.?[0-9]+)", proc.stdout or "")
	if not m:
		return {
			"ok": False,
			"iou": None,
			"error": f"IoU parse failed.\nSTDOUT:\n{_tail(proc.stdout or '')}\n\nSTDERR:\n{_tail(proc.stderr or '')}",
		}
	# Safety clamp in case an unexpected cq_align_shapes implementation is imported.
	iou_val = float(m.group(1))
	if iou_val > 1.0:
		iou_val = 1.0
	if iou_val < 0.0:
		iou_val = 0.0
	return {"ok": True, "iou": iou_val, "error": None}


def _autodetect_image_folder(question_file: str, *, fallback_image_folder: str) -> str:
	"""
	If the first non-empty JSONL row has an absolute `image` path that exists on disk,
	return "" (meaning: do not prepend an image folder). Otherwise return the
	configured image-folder fallback.
	"""
	path = os.path.expanduser(question_file)
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			img = str(obj.get("image", ""))
			if img.startswith("/") and os.path.exists(img):
				return ""
			break
	return fallback_image_folder


def _resolve_image_path(*, image_value: Any, image_folder: str) -> str:
	img = str(image_value)
	# If the image path is absolute, use it directly.
	if os.path.isabs(img):
		if os.path.exists(img):
			return img
		raise FileNotFoundError(img)
	# If the image path is relative, try resolving it within the image folder.
	if image_folder:
		p = os.path.join(image_folder, img)
		if os.path.exists(p):
			return p
		raise FileNotFoundError(p)
	raise FileNotFoundError(
		f"Relative image path '{img}' but image_folder is empty; pass --image-folder or use a JSONL with absolute paths."
	)


def _resolve_gt_step_path(*, step_value: str, gt_step_folder: str) -> str:
	"""Resolve a ground-truth STEP path.

	Accepts:
	- absolute existing path
	- path relative to gt_step_folder
	- STEP id/stem (e.g. '22457_a6c2776f_0008') searched under gt_step_folder
	"""
	v = str(step_value).strip()
	if not v:
		raise FileNotFoundError("Empty step id/path")

	# If it's an absolute path, use it directly.
	if os.path.isabs(v):
		if os.path.exists(v):
			return v
		raise FileNotFoundError(v)

	gt_step_folder = os.path.expanduser(gt_step_folder)
	folder_abs = os.path.abspath(gt_step_folder)

	# If it's a relative path (possibly with subdirs), try resolving within gt_step_folder.
	candidate_rel = os.path.abspath(os.path.normpath(os.path.join(folder_abs, v)))
	if candidate_rel.startswith(folder_abs + os.sep) and os.path.exists(candidate_rel):
		return candidate_rel

	stem = os.path.splitext(os.path.basename(v))[0]
	if not stem:
		raise FileNotFoundError(v)

	# Fast paths: exact filename under gt_step_folder.
	for ext in (".step", ".stp", ".STEP", ".STP"):
		p = os.path.join(folder_abs, stem + ext)
		if os.path.exists(p):
			return p

	# Fallback: search recursively for an exact stem match.
	patterns = [
		os.path.join(folder_abs, "**", stem + ".step"),
		os.path.join(folder_abs, "**", stem + ".stp"),
		os.path.join(folder_abs, "**", stem + ".STEP"),
		os.path.join(folder_abs, "**", stem + ".STP"),
	]
	matches: List[str] = []
	for pat in patterns:
		matches.extend(glob.glob(pat, recursive=True))
	matches = sorted(set(matches))
	if len(matches) == 1:
		return matches[0]
	if len(matches) > 1:
		# Deterministic choice; surface ambiguity to the caller via exception.
		raise FileNotFoundError(f"Multiple GT STEP matches for '{stem}': {matches[:5]}{'...' if len(matches) > 5 else ''}")

	raise FileNotFoundError(f"GT STEP not found for '{v}' under '{folder_abs}'")


def _create_response(
	*,
	req_client,
	model: str,
	input_payload: List[Dict[str, Any]],
	max_output_tokens: int,
):
	kwargs: Dict[str, Any] = {
		"model": model,
		"input": input_payload,
		"max_output_tokens": max_output_tokens,
	}
	return req_client.responses.create(**kwargs)


def _response_output_text(resp) -> str:
	out_text = getattr(resp, "output_text", None)
	if out_text is None:
		out_text = str(resp)
	return _strip_code_fences(str(out_text))


def _anthropic_image_source(image_url: str) -> Dict[str, Any]:
	data_match = re.match(r"^data:([^;]+);base64,(.*)$", image_url, flags=re.DOTALL)
	if data_match:
		media_type = data_match.group(1)
		if media_type == "application/octet-stream":
			raise ValueError("Claude image inputs require a supported image media type, not application/octet-stream.")
		return {"type": "base64", "media_type": media_type, "data": data_match.group(2)}
	raise ValueError("Claude image inputs must be data URLs produced from local image files.")


def _anthropic_content(content: Any) -> Any:
	if isinstance(content, str):
		return content
	if not isinstance(content, list):
		return [{"type": "text", "text": str(content)}]

	anthropic_blocks: List[Dict[str, Any]] = []
	for block in content:
		if not isinstance(block, dict):
			anthropic_blocks.append({"type": "text", "text": str(block)})
			continue
		block_type = block.get("type")
		if block_type in {"input_text", "text"}:
			anthropic_blocks.append({"type": "text", "text": str(block.get("text", ""))})
		elif block_type in {"input_image", "image"}:
			image_url = str(block.get("image_url") or block.get("url") or "")
			anthropic_blocks.append({"type": "image", "source": _anthropic_image_source(image_url)})
		else:
			anthropic_blocks.append({"type": "text", "text": json.dumps(block, ensure_ascii=False)})
	return anthropic_blocks


def _anthropic_messages(input_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	messages: List[Dict[str, Any]] = []
	for message in input_messages:
		role = str(message.get("role") or "user")
		if role not in {"user", "assistant"}:
			role = "user"
		messages.append({"role": role, "content": _anthropic_content(message.get("content", ""))})
	return messages


def _anthropic_output_text(resp) -> str:
	parts: List[str] = []
	for block in getattr(resp, "content", []) or []:
		text = getattr(block, "text", None)
		if text is None and isinstance(block, dict):
			text = block.get("text")
		if text is not None:
			parts.append(str(text))
	if not parts:
		return _strip_code_fences(str(resp))
	return _strip_code_fences("\n".join(parts))


def _create_anthropic_message(
	*,
	req_client,
	model: str,
	input_messages: List[Dict[str, Any]],
	max_output_tokens: int,
):
	kwargs: Dict[str, Any] = {
		"model": model,
		"messages": _anthropic_messages(input_messages),
		"max_tokens": max_output_tokens,
	}
	return req_client.messages.create(**kwargs)


def _call_model_input(
	*,
	client,
	model: str,
	input_payload: List[Dict[str, Any]],
	max_output_tokens: int,
	timeout_s: float,
	max_retries: int,
) -> str:
	provider = client.get("provider", "openai") if isinstance(client, dict) else "openai"
	sdk_client = client.get("client") if isinstance(client, dict) else client
	last_err: Optional[BaseException] = None
	for attempt in range(max_retries + 1):
		try:
			# Set per-request timeout via SDK options (works across SDK versions).
			req_client = sdk_client.with_options(timeout=timeout_s) if hasattr(sdk_client, "with_options") else sdk_client
			if provider == "anthropic":
				resp = _create_anthropic_message(
					req_client=req_client,
					model=model,
					input_messages=input_payload,
					max_output_tokens=max_output_tokens,
				)
				return _anthropic_output_text(resp)
			resp = _create_response(
				req_client=req_client,
				model=model,
				input_payload=input_payload,
				max_output_tokens=max_output_tokens,
			)
			return _response_output_text(resp)
		except Exception as e:
			last_err = e
			if attempt >= max_retries:
				break
			# Basic exponential backoff with cap.
			sleep_s = min(60.0, (2.0**attempt) + 0.25)
			time.sleep(sleep_s)

	raise RuntimeError(f"{provider} call failed after {max_retries + 1} attempts: {last_err}")


def call_model_vision(
	*,
	client,
	model: str,
	prompt: str,
	image_path: str,
	max_output_tokens: int,
	timeout_s: float,
	max_retries: int,
) -> str:
	image_url = _encode_image_as_data_url(image_path)
	return _call_model_input(
		client=client,
		model=model,
		input_payload=[
			{
				"role": "user",
				"content": [
					{"type": "input_image", "image_url": image_url},
					{"type": "input_text", "text": prompt},
				],
			}
		],
		max_output_tokens=max_output_tokens,
		timeout_s=timeout_s,
		max_retries=max_retries,
	)


def call_model_messages(
	*,
	client,
	model: str,
	input_messages: List[Dict[str, Any]],
	max_output_tokens: int,
	timeout_s: float,
	max_retries: int,
) -> str:
	return _call_model_input(
		client=client,
		model=model,
		input_payload=input_messages,
		max_output_tokens=max_output_tokens,
		timeout_s=timeout_s,
		max_retries=max_retries,
	)


def _extract_unified_diff(model_text: str) -> str:
	"""Extract a unified diff from a model response."""
	text = str(model_text or "").strip()
	fence = re.search(r"```(?:diff|patch)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
	if fence:
		text = fence.group(1).strip()

	lines = text.splitlines()
	start_idx: Optional[int] = None
	for i, line in enumerate(lines):
		if line.startswith("--- "):
			for j in range(i + 1, min(i + 4, len(lines))):
				if lines[j].startswith("+++ "):
					start_idx = i
					break
		if start_idx is not None:
			break
	if start_idx is None:
		return ""
	return "\n".join(lines[start_idx:]).rstrip() + "\n"


def _cadquery_patch_applier_code() -> str:
	return r'''
from __future__ import annotations

import os
import difflib
import re
import sys


def _split_keepends(text: str) -> list[str]:
	return text.splitlines(keepends=True)


def _matches_at(lines: list[str], pos: int, old_lines: list[str]) -> bool:
	if pos < 0 or pos + len(old_lines) > len(lines):
		return False
	return lines[pos : pos + len(old_lines)] == old_lines


def _normalized(line: str) -> str:
	return line.rstrip()


def _normalized_matches_at(lines: list[str], pos: int, old_lines: list[str]) -> bool:
	if pos < 0 or pos + len(old_lines) > len(lines):
		return False
	return [_normalized(x) for x in lines[pos : pos + len(old_lines)]] == [_normalized(x) for x in old_lines]


def _find_exact_hunk_position(lines: list[str], old_lines: list[str], expected: int, min_pos: int) -> int | None:
	if _matches_at(lines, expected, old_lines) and expected >= min_pos:
		return expected
	window_start = max(min_pos, expected - 30)
	window_end = min(len(lines) - len(old_lines), expected + 30)
	for pos in range(window_start, window_end + 1):
		if _matches_at(lines, pos, old_lines):
			return pos
	for pos in range(min_pos, len(lines) - len(old_lines) + 1):
		if _matches_at(lines, pos, old_lines):
			return pos
	return None


def _find_normalized_hunk_position(lines: list[str], old_lines: list[str], expected: int, min_pos: int) -> int | None:
	if _normalized_matches_at(lines, expected, old_lines) and expected >= min_pos:
		return expected
	window_start = max(min_pos, expected - 30)
	window_end = min(len(lines) - len(old_lines), expected + 30)
	for pos in range(window_start, window_end + 1):
		if _normalized_matches_at(lines, pos, old_lines):
			return pos
	for pos in range(min_pos, len(lines) - len(old_lines) + 1):
		if _normalized_matches_at(lines, pos, old_lines):
			return pos
	return None


def _find_fuzzy_hunk_position(lines: list[str], old_lines: list[str], expected: int, min_pos: int) -> tuple[int, int] | None:
	if not old_lines:
		return (max(min_pos, expected), 0)
	old_text = "".join(_normalized(x) + "\n" for x in old_lines)
	best: tuple[float, int, int] | None = None
	lengths = range(max(1, len(old_lines) - 3), min(len(lines) - min_pos, len(old_lines) + 3) + 1)
	for length in lengths:
		max_pos = len(lines) - length
		for pos in range(min_pos, max_pos + 1):
			window_text = "".join(_normalized(x) + "\n" for x in lines[pos : pos + length])
			score = difflib.SequenceMatcher(None, old_text, window_text).ratio()
			if best is None or score > best[0]:
				best = (score, pos, length)
	if best is not None and best[0] >= 0.78:
		return (best[1], best[2])
	return None


def apply_unified_diff(original: str, patch: str) -> str:
	orig_lines = _split_keepends(original)
	patch_lines = _split_keepends(patch)
	hunks: list[tuple[int, list[str]]] = []
	i = 0
	while i < len(patch_lines):
		line = patch_lines[i]
		if not line.startswith("@@ "):
			i += 1
			continue
		m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
		if not m:
			raise ValueError(f"Unsupported hunk header: {line.rstrip()}")
		old_start = int(m.group(1))
		i += 1
		body: list[str] = []
		while i < len(patch_lines) and not patch_lines[i].startswith("@@ "):
			if patch_lines[i].startswith(("--- ", "+++ ")):
				break
			if patch_lines[i].startswith("\\ No newline at end of file"):
				i += 1
				continue
			if not patch_lines[i] or patch_lines[i][0] not in " +-":
				raise ValueError(f"Unsupported patch line: {patch_lines[i].rstrip()}")
			body.append(patch_lines[i])
			i += 1
		hunks.append((old_start, body))
	if not hunks:
		raise ValueError("Patch did not contain any unified-diff hunks.")

	output: list[str] = []
	orig_pos = 0
	for old_start, body in hunks:
		old_lines = [line[1:] for line in body if line[0] in " -"]
		new_lines = [line[1:] for line in body if line[0] in " +"]
		target = _find_exact_hunk_position(orig_lines, old_lines, old_start - 1, orig_pos)
		if target is None:
			target = _find_normalized_hunk_position(orig_lines, old_lines, old_start - 1, orig_pos)
		if target is None:
			fuzzy = _find_fuzzy_hunk_position(orig_lines, old_lines, old_start - 1, orig_pos)
			if fuzzy is None:
				raise ValueError("Could not find matching context for patch hunk.")
			target, matched_len = fuzzy
			output.extend(orig_lines[orig_pos:target])
			output.extend(new_lines)
			orig_pos = target + matched_len
			continue
		output.extend(orig_lines[orig_pos:target])
		hunk_pos = target
		for line in body:
			sign = line[0]
			value = line[1:]
			if sign == " ":
				if hunk_pos >= len(orig_lines) or _normalized(orig_lines[hunk_pos]) != _normalized(value):
					raise ValueError("Patch context did not match current code.")
				output.append(orig_lines[hunk_pos])
				hunk_pos += 1
			elif sign == "-":
				if hunk_pos >= len(orig_lines) or _normalized(orig_lines[hunk_pos]) != _normalized(value):
					raise ValueError("Patch deletion did not match current code.")
				hunk_pos += 1
			elif sign == "+":
				output.append(value)
		orig_pos = hunk_pos
	output.extend(orig_lines[orig_pos:])
	return "".join(output)


def main() -> None:
	if len(sys.argv) != 4:
		raise SystemExit("usage: apply_cadquery_patch.py CURRENT.py PATCH.diff OUTPUT.py")
	current_path, patch_path, output_path = sys.argv[1:]
	with open(current_path, "r", encoding="utf-8") as f:
		current = f.read()
	with open(patch_path, "r", encoding="utf-8") as f:
		patch = f.read()
	patched = apply_unified_diff(current, patch)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(patched)


if __name__ == "__main__":
	main()
'''.lstrip()


def apply_model_patch_to_cadquery_code(
	*,
	current_code: str,
	model_patch: str,
	qid: str,
	pass_name: str,
	work_dir: str,
	timeout_s: float = 30.0,
) -> Dict[str, Any]:
	"""Apply a model-produced unified diff to the current CadQuery code."""
	os.makedirs(work_dir, exist_ok=True)
	patch_text = _extract_unified_diff(model_patch)
	current_path = os.path.join(work_dir, f"{qid}.{pass_name}.current_cadquery.py")
	patch_path = os.path.join(work_dir, f"{qid}.{pass_name}.patch.diff")
	output_path = os.path.join(work_dir, f"{qid}.{pass_name}.patched_cadquery.py")
	applier_path = os.path.join(work_dir, "apply_cadquery_patch.py")
	_write_text(current_path, current_code)
	_write_text(patch_path, patch_text or str(model_patch or ""))
	_write_text(applier_path, _cadquery_patch_applier_code())
	if not patch_text:
		return {
			"ok": False,
			"code": current_code,
			"patch": model_patch,
			"patch_path": patch_path,
			"error": "Model response did not contain a unified diff starting with --- and +++ headers.",
		}
	try:
		proc = subprocess.run(
			[sys.executable, applier_path, current_path, patch_path, output_path],
			text=True,
			capture_output=True,
			timeout=timeout_s,
		)
	except subprocess.TimeoutExpired as e:
		return {
			"ok": False,
			"code": current_code,
			"patch": patch_text,
			"patch_path": patch_path,
			"error": f"Patch application timed out after {timeout_s}s\nSTDOUT:\n{_tail(getattr(e, 'stdout', '') or '')}\n\nSTDERR:\n{_tail(getattr(e, 'stderr', '') or '')}",
		}
	if proc.returncode != 0 or not os.path.exists(output_path):
		return {
			"ok": False,
			"code": current_code,
			"patch": patch_text,
			"patch_path": patch_path,
			"error": f"Patch application failed (rc={proc.returncode}).\nSTDOUT:\n{_tail(proc.stdout or '')}\n\nSTDERR:\n{_tail(proc.stderr or '')}",
		}
	with open(output_path, "r", encoding="utf-8") as f:
		patched_code = f.read()
	return {
		"ok": True,
		"code": patched_code,
		"patch": patch_text,
		"patch_path": patch_path,
		"current_path": current_path,
		"output_path": output_path,
		"error": None,
	}


def _cadquery_patch_instructions() -> str:
	return (
		"Return ONLY a unified diff patch for the virtual file current_cadquery.py.\n"
		"The diff must start with:\n"
		"--- current_cadquery.py\n"
		"+++ current_cadquery.py\n"
		"Use enough unchanged context lines so the patch applies exactly.\n"
		"You may include multiple @@ hunks when multiple separate code blocks need changes.\n"
		"Prefer separate hunks for distant edits instead of one very large hunk.\n"
		"Do not include markdown fences, explanations, or a full rewritten file.\n"
		"The patched code must assign the final CadQuery solid to variable 'solid'.\n"
		"Do not add export or visualization code."
	)


def run_patch_refinement_for_row(
	*,
	args: argparse.Namespace,
	client,
	model: str,
	qid: Any,
	prompt: str,
	image_path: str,
	step_dir: str,
	work_dir: str,
	gt_step_value: Optional[str],
	gt_step_path: Optional[str],
	gt_step_resolve_error: Optional[str],
) -> Dict[str, Any]:
	n_loops = 2
	n_loops_error = 5
	step_path = os.path.join(step_dir, f"{qid}.step")

	codes: Dict[str, str] = {}
	exec_results: Dict[str, Dict[str, Any]] = {}
	iou_results: Dict[str, Dict[str, Any]] = {}
	render_results: Dict[str, Dict[str, Any]] = {}
	plan_results: List[Dict[str, Any]] = []
	patch_results: List[Dict[str, Any]] = []
	selection_results: List[Dict[str, Any]] = []
	artifact_records: List[Dict[str, Any]] = []
	final_stage_records: List[Dict[str, Any]] = []
	model_call_history: List[Dict[str, Any]] = []

	def _mk_skipped_iou(reason: str) -> Dict[str, Any]:
		return {"ok": False, "iou": None, "error": reason, "skipped": True}

	def _mk_propagated_iou(value: float, *, from_pass: str) -> Dict[str, Any]:
		return {"ok": True, "iou": float(value), "error": None, "propagated_from": from_pass}

	def _mk_propagated_iou_obj(iou_obj: Optional[Dict[str, Any]], *, from_pass: str) -> Dict[str, Any]:
		v = _iou_value(iou_obj)
		if v is not None:
			return _mk_propagated_iou(v, from_pass=from_pass)
		out = _mk_skipped_iou(f"no_iou_to_propagate_from_{from_pass}")
		out["propagated_from"] = from_pass
		return out

	def _fill_unattempted_loop_ious(start_loop_idx: int, reason: str) -> None:
		for remaining_loop_idx in range(start_loop_idx, n_loops + 1):
			pass_name = f"loop{remaining_loop_idx}"
			iou_results.setdefault(pass_name, _mk_skipped_iou(reason))
			iou_results.setdefault(f"{pass_name}_final", _mk_skipped_iou(reason))

	def _iou_value(iou_obj: Optional[Dict[str, Any]]) -> Optional[float]:
		if not isinstance(iou_obj, dict) or not bool(iou_obj.get("ok")):
			return None
		v = iou_obj.get("iou")
		if v is None:
			return None
		try:
			return float(v)
		except Exception:
			return None

	def _logged_iou_value(iou_obj: Optional[Dict[str, Any]]) -> Optional[float]:
		if not isinstance(iou_obj, dict):
			return None
		v = iou_obj.get("iou")
		if v is None:
			return None
		try:
			return float(v)
		except Exception:
			return None

	def _compute_iou_for_step(step_for_iou: str) -> Dict[str, Any]:
		if gt_step_path and os.path.exists(gt_step_path):
			return compute_step_iou(
				conda_env=args.conda_env_cq,
				pred_step_path=step_for_iou,
				gt_step_path=gt_step_path,
				timeout_s=args.iou_timeout_s,
			)
		return _mk_skipped_iou("gt_step_missing")

	def _code_artifact_path(pass_name: str) -> str:
		return os.path.join(step_dir, "cadquery_code", f"{qid}.{pass_name}.py")

	def _step_artifact_path(pass_name: str) -> str:
		return os.path.join(step_dir, f"{qid}.{pass_name}.step")

	def _ortho_svg_artifact_path(pass_name: str) -> str:
		return os.path.join(step_dir, "ortho_svg", f"{qid}.{pass_name}.svg")

	def _ortho_png_artifact_path(pass_name: str) -> str:
		return os.path.join(step_dir, "ortho_png", f"{qid}.{pass_name}.png")

	def _record_artifacts(pass_name: str, code: str, exec_result: Dict[str, Any]) -> Dict[str, Any]:
		code_path = _code_artifact_path(pass_name)
		_write_text(code_path, code)
		record: Dict[str, Any] = {
			"pass_name": pass_name,
			"code_path": code_path,
			"step_path": None,
			"ortho_png": None,
			"ortho_svg": None,
			"iou": None,
			"exec_ok": bool(exec_result.get("ok")),
		}
		if not exec_result.get("ok"):
			iou_results[pass_name] = _mk_skipped_iou(f"{pass_name}_failed")
			record["iou"] = iou_results[pass_name]
			artifact_records.append(record)
			return record

		stable_step_path = _step_artifact_path(pass_name)
		try:
			os.makedirs(os.path.dirname(stable_step_path), exist_ok=True)
			shutil.copy2(step_path, stable_step_path)
			record["step_path"] = stable_step_path
		except Exception as e:
			record["step_copy_error"] = f"Failed to copy STEP for {pass_name}: {type(e).__name__}: {e}"
			stable_step_path = step_path
			record["step_path"] = step_path

		iou_results[pass_name] = _compute_iou_for_step(stable_step_path)
		record["iou"] = iou_results[pass_name]

		ortho_svg = _ortho_svg_artifact_path(pass_name)
		ortho_png = _ortho_png_artifact_path(pass_name)
		render_result = render_step_to_ortho_png(
			conda_env=args.conda_env_occ,
			step_path=stable_step_path,
			svg_path=ortho_svg,
			png_path=ortho_png,
			timeout_s=args.occ_timeout_s_total,
			occ_timeout_s=args.occ_timeout_s,
		)
		render_results[pass_name] = render_result
		if render_result.get("ok"):
			record["ortho_svg"] = ortho_svg
			record["ortho_png"] = ortho_png
		else:
			render_error = str(render_result.get("error") or "")
			record["ortho_error"] = render_error
			iou_results[pass_name] = _mk_skipped_iou(f"{pass_name}_ortho_png_missing: {render_error}")
			record["iou"] = iou_results[pass_name]
		artifact_records.append(record)
		return record

	def _latest_successful_record() -> Optional[Dict[str, Any]]:
		for record in reversed(artifact_records):
			if record.get("exec_ok") and record.get("step_path"):
				return record
		return None

	def _final_stage_record(stage_pass_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
		out = dict(record)
		out["stage_pass_name"] = stage_pass_name
		out["source_pass_name"] = record.get("pass_name")
		return out

	def _call_error_patch(pass_name: str, code: str, error_text: str) -> str:
		error_prompt = (
			"You are helping debug CadQuery code.\n"
			"The current CadQuery code failed to execute and export a STEP file.\n"
			"Create a patch that edits the current CadQuery code to resolve the execution error.\n\n"
			"Original prompt:\n"
			f"{prompt}\n\n"
			"Execution error (verbatim):\n"
			f"{error_text}\n\n"
			"Current CadQuery code in current_cadquery.py:\n"
			"```python\n"
			f"{code}\n"
			"```\n\n"
			f"{_cadquery_patch_instructions()}"
		)
		error_message = {
			"role": "user",
			"content": [
				{
					"type": "input_image",
					"image_url": _encode_image_as_data_url(image_path),
				},
				{"type": "input_text", "text": error_prompt},
			],
		}
		patch_text = call_model_messages(
			client=client,
			model=model,
			input_messages=[*model_call_history, error_message],
			max_output_tokens=args.max_output_tokens,
			timeout_s=args.timeout_s,
			max_retries=args.max_retries,
		)
		model_call_history.extend([error_message, {"role": "assistant", "content": patch_text}])
		return patch_text

	def _repair_execution_errors(base_pass_name: str, code: str, exec_result: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
		current_code = code
		current_exec = exec_result
		final_pass_name = base_pass_name
		for error_idx in range(1, n_loops_error + 1):
			if current_exec.get("ok"):
				break
			error_text = str(current_exec.get("error") or "")
			repair_pass_name = f"{base_pass_name}_error{error_idx}"
			patch_text = _call_error_patch(repair_pass_name, current_code, error_text)
			apply_result = apply_model_patch_to_cadquery_code(
				current_code=current_code,
				model_patch=patch_text,
				qid=str(qid),
				pass_name=repair_pass_name,
				work_dir=work_dir,
			)
			patch_results.append({"pass_name": repair_pass_name, "kind": "error_fix", "apply": apply_result})
			if not apply_result.get("ok"):
				current_exec = {
					"ok": False,
					"stdout": "",
					"stderr": "",
					"error": str(apply_result.get("error") or "Patch application failed."),
				}
				exec_results[repair_pass_name] = current_exec
				continue
			current_code = str(apply_result.get("code") or current_code)
			codes[repair_pass_name] = current_code
			current_exec = execute_cadquery_to_step(
				conda_env=args.conda_env_cq,
				qid=str(qid),
				pass_name=repair_pass_name,
				cadquery_code=current_code,
				step_path=step_path,
				work_dir=work_dir,
				timeout_s=args.cq_timeout_s,
			)
			exec_results[repair_pass_name] = current_exec
			final_pass_name = repair_pass_name
		return current_code, current_exec, final_pass_name

	def _visual_plan_prompt(code: str) -> str:
		return (
			"You are helping debug CadQuery code.\n"
			"You will be given two orthographic projection images:\n"
			"1) Ground-truth projection (correct).\n"
			"2) Projection rendered from the current CadQuery code's exported STEP (incorrect).\n\n"
			"Compare the ground-truth geometry in the ground-truth projection and the geometry in the current projection: "
			"outer silhouettes, holes, cutouts, bosses, feature sizes, feature positions relative to one another, and overall proportions. "
			"The orientation of the projected geometry does not matter. Do not penalize rotation, reflection, or image placement if the "
			"underlying projected part geometry matches the ground-truth geometry.\n\n"
			"List the similarities and differences between the two geometries. Then output a structured plan for changing the current CadQuery code to fix these differences.\n"
			"Do not output CadQuery code yet.\n\n"
			"Original prompt:\n"
			f"{prompt}\n\n"
			"Current CadQuery code in current_cadquery.py:\n"
			"```python\n"
			f"{code}\n"
			"```"
		)

	def _visual_rewrite_prompt(code: str) -> str:
		return (
			"Using the structured plan above, output only the corrected and error-free executable CadQuery code "
			"so that the exported STEP's geometry matches the ground-truth geometry. "
			"Assign the final solid to variable 'solid' in the last line. Do not export or visualize.\n\n"
			"Current CadQuery code in current_cadquery.py:\n"
			"```python\n"
			f"{code}\n"
			"```"
		)

	def _call_visual_plan(code: str, pred_ortho_png: str) -> str:
		gt_image_url = _encode_image_as_data_url(image_path)
		pred_image_url = _encode_image_as_data_url(pred_ortho_png)
		plan_message = {
			"role": "user",
			"content": [
				{"type": "input_image", "image_url": gt_image_url},
				{"type": "input_image", "image_url": pred_image_url},
				{"type": "input_text", "text": _visual_plan_prompt(code)},
			],
		}
		plan_text = call_model_messages(
			client=client,
			model=model,
			input_messages=[*model_call_history, plan_message],
			max_output_tokens=args.max_output_tokens,
			timeout_s=args.timeout_s,
			max_retries=args.max_retries,
		)
		plan_text = plan_text.strip()
		model_call_history.extend([plan_message, {"role": "assistant", "content": plan_text}])
		return plan_text

	def _call_visual_rewrite(code: str) -> str:
		rewrite_message = {"role": "user", "content": _visual_rewrite_prompt(code)}
		rewrite_text = call_model_messages(
			client=client,
			model=model,
			input_messages=[*model_call_history, rewrite_message],
			max_output_tokens=args.max_output_tokens,
			timeout_s=args.timeout_s,
			max_retries=args.max_retries,
		)
		rewrite_text = _strip_code_fences(rewrite_text)
		model_call_history.extend([rewrite_message, {"role": "assistant", "content": rewrite_text}])
		return rewrite_text

	def _choose_best_overall_artifact(records: List[Dict[str, Any]]) -> Dict[str, Any]:
		selectable = [
			record for record in records
			if record.get("exec_ok") and record.get("ortho_png") and os.path.exists(str(record.get("ortho_png")))
		]
		if not selectable:
			return records[-1] if records else {}

		gt_image_url = _encode_image_as_data_url(image_path)
		content: List[Dict[str, Any]] = [{"type": "input_image", "image_url": gt_image_url}]
		mapping: List[str] = []
		for idx, record in enumerate(selectable, start=1):
			content.append({
				"type": "input_image",
				"image_url": _encode_image_as_data_url(str(record.get("ortho_png"))),
			})
			stage_pass_name = record.get("stage_pass_name") or record.get("pass_name")
			source_pass_name = record.get("source_pass_name") or record.get("pass_name")
			if stage_pass_name != source_pass_name:
				mapping.append(f"candidate {idx}: {stage_pass_name} is image {idx + 1} (rendered from {source_pass_name})")
			else:
				mapping.append(f"candidate {idx}: {stage_pass_name} is image {idx + 1}")

		score_options = "\n".join(f"SCORE {idx}: <integer from 0 to 100>" for idx in range(1, len(selectable) + 1))
		selection_prompt = (
			"You are selecting the best final CadQuery output.\n"
			"The first image is the ground-truth projection. Each following image is a projection rendered from "
			"one CadQuery code output generated during this run, from exec0 through the final loop.\n\n"
			"Compare each rendered projection with the ground-truth projection by focusing on the geometry of the part being projected "
			"and the ground-truth geometry shown in the ground-truth projection: outer silhouettes, holes, cutouts, bosses, feature sizes, "
			"feature positions relative to one another, and overall proportions. The orientation of the projected geometry does not matter. "
			"Do not penalize rotation, reflection, or image placement if the underlying projected part geometry matches the ground-truth geometry.\n\n"
			"Give every candidate a geometry match score from 0 to 100, where 100 means a perfect geometry match ignoring orientation. "
			"Then select the candidate with the highest score.\n\n"
			"Reply with one score line per candidate, followed by the best line, exactly like this:\n"
			f"{score_options}\n"
			"BEST: <candidate number with the highest score>\n\n"
			"Then provide a short reason after those lines. Judge only from the images and chat history.\n\n"
			"Candidate mapping:\n"
			+ "\n".join(mapping)
		)
		content.append({"type": "input_text", "text": selection_prompt})
		selection_message = {"role": "user", "content": content}
		selection_text = call_model_messages(
			client=client,
			model=model,
			input_messages=[*model_call_history, selection_message],
			max_output_tokens=args.max_output_tokens,
			timeout_s=args.timeout_s,
			max_retries=args.max_retries,
		).strip()
		model_call_history.extend([selection_message, {"role": "assistant", "content": selection_text}])

		scores: Dict[str, Optional[float]] = {}
		for idx in range(1, len(selectable) + 1):
			score_match = re.search(rf"SCORE\s+{idx}\s*:\s*(\d+(?:\.\d+)?)", selection_text, flags=re.IGNORECASE)
			score_value: Optional[float] = None
			if score_match:
				try:
					score_value = max(0.0, min(100.0, float(score_match.group(1))))
				except Exception:
					score_value = None
			scores[str(idx)] = score_value
		scored_indices = [(idx - 1, score) for idx, score in ((i, scores[str(i)]) for i in range(1, len(selectable) + 1)) if score is not None]
		best_by_score = max(scored_indices, key=lambda x: x[1])[0] if scored_indices else 0
		m = re.search(r"BEST:\s*(\d+)", selection_text, flags=re.IGNORECASE)
		chosen_idx = int(m.group(1)) - 1 if m else best_by_score
		if chosen_idx < 0 or chosen_idx >= len(selectable):
			chosen_idx = best_by_score
		chosen = selectable[chosen_idx]
		selection_results.append({
			"pass_name": "final",
			"kind": "overall_model_selection",
			"chosen_pass_name": chosen.get("pass_name"),
			"chosen_stage_pass_name": chosen.get("stage_pass_name"),
			"chosen_index": chosen_idx + 1,
			"scores": scores,
			"selection_text": selection_text,
			"candidates": [record.get("stage_pass_name") or record.get("pass_name") for record in selectable],
			"candidate_source_passes": [record.get("source_pass_name") or record.get("pass_name") for record in selectable],
		})
		return chosen

	codes["exec0"] = _strip_code_fences(
		call_model_vision(
			client=client,
			model=model,
			prompt=prompt,
			image_path=image_path,
			max_output_tokens=args.max_output_tokens,
			timeout_s=args.timeout_s,
			max_retries=args.max_retries,
		)
	)
	model_call_history.extend([
		{
			"role": "user",
			"content": [
				{
					"type": "input_image",
					"image_url": _encode_image_as_data_url(image_path),
				},
				{"type": "input_text", "text": prompt},
			],
		},
		{"role": "assistant", "content": codes["exec0"]},
	])
	current_code = codes["exec0"]
	current_exec = execute_cadquery_to_step(
		conda_env=args.conda_env_cq,
		qid=str(qid),
		pass_name="exec0",
		cadquery_code=current_code,
		step_path=step_path,
		work_dir=work_dir,
		timeout_s=args.cq_timeout_s,
	)
	exec_results["exec0"] = current_exec
	exec0_record = _record_artifacts("exec0", current_code, current_exec)

	if not current_exec.get("ok"):
		current_code, current_exec, exec0_repair_pass_name = _repair_execution_errors("exec0", current_code, current_exec)
		if current_exec.get("ok"):
			codes[exec0_repair_pass_name] = current_code
			exec0_repaired_record = _record_artifacts(exec0_repair_pass_name, current_code, current_exec)
			final_stage_records.append(_final_stage_record("exec0", exec0_repaired_record))
			iou_results["exec0_final"] = _mk_propagated_iou_obj(exec0_repaired_record.get("iou"), from_pass=exec0_repair_pass_name)
		else:
			iou_results["exec0_final"] = _mk_skipped_iou(f"{exec0_repair_pass_name}_failed")
	else:
		final_stage_records.append(_final_stage_record("exec0", exec0_record))
		iou_results["exec0_final"] = _mk_propagated_iou_obj(iou_results.get("exec0"), from_pass="exec0")

	for loop_idx in range(1, n_loops + 1):
		if not current_exec.get("ok"):
			_fill_unattempted_loop_ious(loop_idx, "previous_execution_failed")
			break
		latest_record = _latest_successful_record()
		pred_ortho_png = str((latest_record or {}).get("ortho_png") or "")
		if not pred_ortho_png or not os.path.exists(pred_ortho_png):
			reason = "No current orthographic PNG available for visual rewrite prompt."
			render_results[f"loop{loop_idx}"] = {"ok": False, "error": reason}
			_fill_unattempted_loop_ious(loop_idx, reason)
			break

		pass_name = f"loop{loop_idx}"
		plan_text = _call_visual_plan(current_code, pred_ortho_png)
		plan_results.append({
			"pass_name": pass_name,
			"kind": "visual_compare_plan",
			"source_ortho_png": pred_ortho_png,
			"plan": plan_text,
		})
		rewrite_code = _call_visual_rewrite(current_code)
		apply_result = {
			"ok": bool(rewrite_code.strip()),
			"code": rewrite_code if rewrite_code.strip() else current_code,
			"mode": "full_rewrite",
			"error": "" if rewrite_code.strip() else "Model returned empty rewritten CadQuery code.",
		}
		patch_results.append({
			"pass_name": pass_name,
			"kind": "visual_refine",
			"source_ortho_png": pred_ortho_png,
			"plan": plan_text,
			"apply": apply_result,
		})
		if apply_result.get("ok"):
			current_code = str(apply_result.get("code") or current_code)
			codes[pass_name] = current_code
			current_exec = execute_cadquery_to_step(
				conda_env=args.conda_env_cq,
				qid=str(qid),
				pass_name=pass_name,
				cadquery_code=current_code,
				step_path=step_path,
				work_dir=work_dir,
				timeout_s=args.cq_timeout_s,
			)
		else:
			current_exec = {
				"ok": False,
				"stdout": "",
				"stderr": "",
				"error": str(apply_result.get("error") or "Visual rewrite failed."),
			}
		exec_results[pass_name] = current_exec
		raw_loop_record = _record_artifacts(pass_name, current_code, current_exec)

		if not current_exec.get("ok"):
			current_code, current_exec, repair_pass_name = _repair_execution_errors(pass_name, current_code, current_exec)
			if current_exec.get("ok"):
				repaired_record = _record_artifacts(repair_pass_name, current_code, current_exec)
				final_stage_records.append(_final_stage_record(pass_name, repaired_record))
				iou_results[f"{pass_name}_final"] = _mk_propagated_iou_obj(repaired_record.get("iou"), from_pass=repair_pass_name)
			else:
				iou_results[f"{pass_name}_final"] = _mk_skipped_iou(f"{repair_pass_name}_failed")
		else:
			final_stage_records.append(_final_stage_record(pass_name, raw_loop_record))
			iou_results[f"{pass_name}_final"] = _mk_propagated_iou_obj(raw_loop_record.get("iou"), from_pass=pass_name)

	successful_records = [record for record in artifact_records if record.get("exec_ok")]
	preliminary_final_record = successful_records[-1] if successful_records else (artifact_records[-1] if artifact_records else {})
	final_record = _choose_best_overall_artifact(final_stage_records) if final_stage_records else preliminary_final_record
	final_exec_name = str(final_record.get("pass_name") or "exec0")
	final_exec_reason = "overall_model_selected_projection" if successful_records else "no_successful_exec"
	final_iou_obj = final_record.get("iou") if isinstance(final_record, dict) else None
	final_iou_value = _logged_iou_value(final_iou_obj if isinstance(final_iou_obj, dict) else None)
	ortho_svg_final = final_record.get("ortho_svg") if isinstance(final_record, dict) else None
	ortho_png_final = final_record.get("ortho_png") if isinstance(final_record, dict) else None
	ortho_final_error = final_record.get("ortho_error") if isinstance(final_record, dict) else None
	final_code = codes.get(final_exec_name, current_code)

	metadata: Dict[str, Any] = {
		"step_dir": step_dir,
		"pred_step_path": step_path,
		"gt_step_value": gt_step_value,
		"gt_step_path": gt_step_path,
		"gt_step_resolve_error": gt_step_resolve_error,
		"n_loops": n_loops,
		"n_loops_error": n_loops_error,
		"final_exec": final_exec_name,
		"final_exec_reason": final_exec_reason,
		"final_iou": final_iou_value,
		"ortho_svg_final": ortho_svg_final,
		"ortho_png_final": ortho_png_final,
		"ortho_final_error": ortho_final_error,
		"artifact_records": artifact_records,
		"plan_results": plan_results,
		"patch_results": patch_results,
		"selection_results": selection_results,
	}

	log_obj: Dict[str, Any] = {
		"question_id": qid,
		"image": image_path,
		"prompt": prompt,
		"gt_step": gt_step_path,
		"gt_step_value": gt_step_value,
		"gt_step_resolve_error": gt_step_resolve_error,
		"step_path": step_path,
		"codes": codes,
		"execs": exec_results,
		"ious": iou_results,
		"renders": render_results,
		"plans": plan_results,
		"patches": patch_results,
		"selections": selection_results,
		"artifact_records": artifact_records,
		"n_loops": n_loops,
		"n_loops_error": n_loops_error,
		"final_exec": final_exec_name,
		"final_exec_reason": final_exec_reason,
		"final_iou": final_iou_value,
		"ortho_svg_final": ortho_svg_final,
		"ortho_png_final": ortho_png_final,
		"ortho_final_error": ortho_final_error,
	}
	output_base_dir = os.getcwd()
	metadata_for_output = _make_paths_relative_for_output(metadata, base_dir=output_base_dir)
	log_obj_for_output = _make_paths_relative_for_output(log_obj, base_dir=output_base_dir)
	_write_json(os.path.join(step_dir, "logs", f"{qid}.json"), log_obj_for_output)

	return {"final_code": final_code, "metadata": metadata_for_output, "log": log_obj_for_output}


def eval_api(args: argparse.Namespace) -> None:
	model = args.model
	provider = _infer_api_provider(model, getattr(args, "api_provider", "auto"))

	image_folder = args.image_folder
	if image_folder == "auto":
		image_folder = _autodetect_image_folder(
			args.question_file,
			fallback_image_folder=args.default_image_folder,
		)

	questions = _read_jsonl(args.question_file)

	answers_file = os.path.expanduser(args.answers_file)
	os.makedirs(os.path.dirname(answers_file), exist_ok=True)

	step_dir = os.path.expanduser(args.step_dir)
	os.makedirs(step_dir, exist_ok=True)
	work_dir = os.path.join(step_dir, "_work")
	os.makedirs(work_dir, exist_ok=True)

	client = _get_api_client(provider)

	with open(answers_file, "w", encoding="utf-8") as out_f:
		for row in tqdm(questions, total=len(questions)):
			qid = row.get("question_id")
			image_file = row.get("image")
			prompt = row.get("text")
			gt_step_value = row.get("step")

			if qid is None or image_file is None or prompt is None:
				raise ValueError(f"Missing required keys in row: {row.keys()}")
			if not isinstance(prompt, str):
				raise ValueError(f"Expected 'text' to be str, got {type(prompt)}")
			if gt_step_value is not None and not isinstance(gt_step_value, str):
				raise ValueError(f"Expected 'step' to be str when present, got {type(gt_step_value)}")

			image_path = _resolve_image_path(image_value=image_file, image_folder=image_folder)

			gt_step_path: Optional[str] = None
			gt_step_resolve_error: Optional[str] = None
			if gt_step_value:
				try:
					gt_step_path = _resolve_gt_step_path(step_value=gt_step_value, gt_step_folder=args.gt_step_folder)
				except Exception as e:
					gt_step_resolve_error = f"Failed to resolve GT STEP '{gt_step_value}': {type(e).__name__}: {e}"
					gt_step_path = None

			patch_pipeline_result = run_patch_refinement_for_row(
				args=args,
				client=client,
				model=model,
				qid=qid,
				prompt=prompt,
				image_path=image_path,
				step_dir=step_dir,
				work_dir=work_dir,
				gt_step_value=gt_step_value,
				gt_step_path=gt_step_path,
				gt_step_resolve_error=gt_step_resolve_error,
			)
			out_f.write(
				json.dumps(
					{
						"question_id": qid,
						"prompt": prompt,
						"text": patch_pipeline_result["final_code"],
						"answer_id": uuid.uuid4().hex,
						"model_id": model,
						"metadata": patch_pipeline_result["metadata"],
					},
					ensure_ascii=False,
				)
				+ "\n"
			)
			out_f.flush()



def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate a vision-capable OpenAI or Anthropic model on a CadQuery JSONL dataset, "
		)
	)
	parser.add_argument("--model", type=str, default="gpt-5.5", help="Model name, e.g. gpt-5.5 or claude-opus-4-8")
	# parser.add_argument("--model", type=str, default="claude-opus-4-8", help="Model name, e.g. gpt-5.5 or claude-opus-4-8")
	parser.add_argument(
		"--api-provider",
		type=str,
		default="auto",
		choices=["auto", "openai", "anthropic"],
		help="API provider. 'auto' selects Anthropic for model names starting with 'claude', otherwise OpenAI.",
	)
	parser.add_argument(
		"--image-folder",
		type=str,
		default="../inference/test100_images_f360_mdim",
		# default="../inference/test100_images_deepcad",
		# default="../inference/test100_images_zerotocad1m",
		help=(
			"Image folder for legacy datasets. Use 'auto' to avoid prepending a folder when JSONL `image` "
			"values are absolute existing paths; otherwise falls back to --default-image-folder."
		),
	)
	parser.add_argument(
		"--default-image-folder",
		type=str,
		default="../inference/test100_images_f360_mdim",
		# default="../inference/test100_images_deepcad",
		# default="../inference/test100_images_zerotocad1m",
		help="Fallback image folder used when --image-folder auto does not find absolute image paths in the JSONL.",
	)
	parser.add_argument(
		"--question-file",
		type=str,
		default="../inference/f360rec_test_data_subset100_mdim_api.jsonl",
		# default="../inference/deepcad_test_data_subset100_api.jsonl",
		# default="../inference/zerotocad1m_test_data_subset100_api.jsonl",
		help="Input JSONL with fields: question_id, image, text.",
	)
	parser.add_argument(
		"--answers-file",
		type=str,
		# default="../inference/inference_results/gpt-5.5/zerotocad1m_test_data_subset100_api_rewrite_2loops/merge.jsonl",
		default="../inference/inference_results/claude-opus-4-8/f360rec_test_data_subset100_mdim_api_rewrite_2loops/merge.jsonl",
		help="Output JSONL path (write merge.jsonl directly).",
	)
	parser.add_argument(
		"--step-dir",
		type=str,
		# default="../inference/inference_results/gpt-5.5/zerotocad1m_test_data_subset100_api_rewrite_2loops",
		default="../inference/inference_results/claude-opus-4-8/f360rec_test_data_subset100_mdim_api_rewrite_2loops",
		help="Directory to write STEP exports + orthographic renderings.",
	)
	parser.add_argument(
		"--gt-step-folder",
		type=str,
		default="../inference/test100_gt_steps_f360",
		# default="../inference/test100_gt_steps_deepcad",
		# default="../inference/test100_gt_steps_zerotocad1m",
		help="Folder containing ground-truth STEP files (looked up by id/stem when JSONL only provides an id).",
	)

	parser.add_argument(
		"--max-output-tokens",
		dest="max_output_tokens",
		type=int,
		default=8192,
		help="Max output tokens (default: 8192)",
	)

	parser.add_argument("--timeout-s", type=float, default=180.0, help="Per-request timeout in seconds.")
	parser.add_argument("--max-retries", type=int, default=6, help="Retries on transient API errors.")

	parser.add_argument(
		"--conda-env-cq",
		type=str,
		default="cad_iou",
		help="Conda env used to execute CadQuery code + compute IoU.",
	)
	parser.add_argument(
		"--conda-env-occ",
		type=str,
		default="pyocc",
		help="Conda env used to render STEP->orthographic (pythonocc) and SVG->PNG (cairosvg).",
	)
	parser.add_argument("--cq-timeout-s", type=float, default=60.0, help="Timeout for executing CadQuery code + exporting STEP.")
	parser.add_argument("--occ-timeout-s", type=float, default=60.0, help="Hard timeout per STEP for pythonocc processing (seconds).")
	parser.add_argument("--occ-timeout-s-total", type=float, default=120.0, help="Total timeout for STEP->SVG and SVG->PNG steps.")
	parser.add_argument("--iou-timeout-s", type=float, default=120.0, help="Timeout for IoU computation.")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	eval_api(args)


if __name__ == "__main__":
	main()
