# GRPO training for Qwen VL with CAD IoU reward

import os
import sys
import copy
import itertools
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import transformers
from transformers import AutoProcessor, TrainerCallback
from qwenvl.data.data_processor import make_supervised_data_module, update_processor_pixels
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import Qwen3VLForConditionalGeneration

from trl import GRPOConfig, GRPOTrainer
import subprocess
import tempfile
import json
import re
import math
import uuid
import shutil
from qwenvl.train.trainer import replace_qwen2_vl_attention_class

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from tqdm import tqdm


def _resolve_image_path(image_root: str, image_rel: str) -> str:
    """Resolve dataset image paths robustly.

    In this project, annotation items often store `image` like `ortho/xxxx/yyyy.png`
    while `data_path` is `/.../ortho_train_data`. This join is correct.
    If `data_path` is already `/.../ortho_train_data/ortho`, this avoids `.../ortho/ortho/...`.
    """
    if not image_rel:
        raise ValueError("Missing image path in sample")

    if os.path.isabs(image_rel) and os.path.exists(image_rel):
        return image_rel

    candidate = os.path.join(image_root, image_rel)
    if os.path.exists(candidate):
        return candidate

    image_root_base = os.path.basename(os.path.normpath(image_root))
    normalized = str(image_rel).replace("\\", "/")
    if image_root_base and normalized.startswith(image_root_base + "/"):
        stripped = normalized[len(image_root_base) + 1 :]
        candidate2 = os.path.join(image_root, stripped)
        if os.path.exists(candidate2):
            return candidate2

    raise FileNotFoundError(
        f"Image not found for sample. Tried: {candidate}"
        + (f", {candidate2}" if 'candidate2' in locals() else "")
    )


def _resolve_step_path(data_root: str, step_path: str) -> str:
    """Resolve a ground-truth STEP file path.

    f360rec annotations typically store absolute STEP paths. This also supports
    relative paths by joining with `data_root`.
    """

    if not step_path:
        raise ValueError("Missing STEP path in sample")

    if os.path.isabs(step_path) and os.path.exists(step_path):
        return step_path

    candidate = os.path.join(data_root, step_path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"STEP file not found for sample. Tried: {step_path}, {candidate}")


IOU_TIMEOUT_SECONDS = float(os.environ.get("CAD_IOU_TIMEOUT", "30"))
NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "8"))
MAX_COMPLETION_TOKENS = int(os.environ.get("GRPO_MAX_COMPLETION_TOKENS", "7680"))  # 8192 context - 229 input tokens - buffer 
LOG_EVERY_N_STEPS = int(os.environ.get("GRPO_LOG_EVERY_N", "10"))
CODE_LOG_DIR = None
CODE_LOG_COUNTER = itertools.count()
LAST_LOGGED_STEP = None
LAST_LOGGED_STEP_CODEPAIR = None
LOGGED_CODEPAIR_IDS = set()
REWARD_TOKENIZER = None

# Optional sanity checks for GRPO reward alignment.
REWARD_DEBUG_HAS_RUN = False


def _env_flag(name: str, default: str = "0") -> bool:
    val = str(os.environ.get(name, default)).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _debug_check_grpo_grouping(
    *,
    sample_ids: list,
    num_generations: int,
    step_index: int | None,
    max_report: int = 5,
) -> None:
    """Best-effort check that GRPO grouping matches TRL's `view(-1, num_generations)` assumption.

    TRL gathers rewards across ranks, then reshapes as (-1, num_generations) to normalize per prompt-group.
    This check gathers `sample_id` across ranks in the same rank-concatenated order and verifies that
    each consecutive block of size `num_generations` has a constant `sample_id`.
    """

    if num_generations <= 0:
        return

    # Only rank 0 prints to avoid log spam.
    rank = 0
    world_size = 1
    if _dist_is_initialized():
        try:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        except Exception:
            rank, world_size = 0, 1

    # Gather sample_ids across ranks in rank order, mimicking TRL's gather() concatenation.
    local_ids = [str(x) for x in (sample_ids or [])]
    if _dist_is_initialized():
        gathered: list[list[str]] = [[] for _ in range(world_size)]
        try:
            torch.distributed.all_gather_object(gathered, local_ids)
        except Exception:
            gathered = [local_ids]
        global_ids = [x for sub in gathered for x in (sub or [])]
    else:
        global_ids = local_ids

    if rank != 0:
        return

    if not global_ids:
        print(f"[grpo_reward_debug] step={step_index} empty sample_ids")
        return

    n = len(global_ids)
    if n % num_generations != 0:
        print(
            f"[grpo_reward_debug] step={step_index} WARNING: total_items={n} not divisible by num_generations={num_generations}. "
            "Grouping/normalization may be misaligned."
        )

    bad_groups = []
    for start in range(0, n - (n % num_generations), num_generations):
        block = global_ids[start : start + num_generations]
        if not block:
            continue
        if len(set(block)) != 1:
            bad_groups.append((start, block))
            if len(bad_groups) >= max_report:
                break

    if bad_groups:
        print(
            f"[grpo_reward_debug] step={step_index} WARNING: found {len(bad_groups)} bad groups (showing up to {max_report}). "
            f"Expected consecutive blocks of {num_generations} to share the same sample_id."
        )
        for start, block in bad_groups:
            print(f"[grpo_reward_debug] group_start={start} sample_ids={block}")
    else:
        # Also print a small summary of counts to confirm distribution.
        from collections import Counter

        counts = Counter(global_ids)
        unique = len(counts)
        top = ", ".join([f"{k}:{v}" for k, v in counts.most_common(5)])
        print(
            f"[grpo_reward_debug] step={step_index} OK: {n} items, {unique} unique sample_ids, "
            f"num_generations={num_generations}. Top counts: {top}"
        )


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_barrier():
    if _dist_is_initialized():
        torch.distributed.barrier()


def _run_bash(command: str, *, cwd: str | None = None, env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a bash command, capturing output for logging/debugging."""
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_cad_iou_results(path: str) -> dict:
    metrics: dict[str, object] = {"avg_iou": None, "valid_steps": None, "raw_text": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        metrics["raw_text"] = raw
        for line in raw.splitlines():
            line = line.strip()
            if line.lower().startswith("average iou"):
                # e.g. "Average IoU: 0.123"
                try:
                    metrics["avg_iou"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            if line.lower().startswith("number of valid steps"):
                try:
                    metrics["valid_steps"] = int(float(line.split(":", 1)[1].strip()))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return metrics


class PeriodicCadIoUEvalCallback(TrainerCallback):
    """After each checkpoint save, run test inference + CAD IoU eval and log to TensorBoard."""

    def __init__(
        self,
        *,
        training_args: TrainingArguments,
        processor=None,
        test_set_name: str = "f360rec_test_data_subset100",
        src_dir: str = "./",
        conda_sh: str = "../../miniconda3/etc/profile.d/conda.sh",
        use_inprocess_test: bool = True,
        limit_eval_to_gpu0: bool = False,
        question_file: str | None = None,
        image_folder: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 8192,
        eval_timeout_seconds: float = 6 * 60 * 60,
    ):
        self.training_args = training_args
        self.processor = processor
        self.test_set_name = test_set_name
        self.src_dir = src_dir
        self.conda_sh = conda_sh
        self.use_inprocess_test = use_inprocess_test
        self.limit_eval_to_gpu0 = limit_eval_to_gpu0
        self.question_file = question_file or f"../inference/{test_set_name}.jsonl"
        self.image_folder = image_folder or "../inference/test100_images"
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.eval_timeout_seconds = eval_timeout_seconds
        self._writer = None

    def _get_dist_info(self) -> tuple[int, int]:
        if _dist_is_initialized():
            return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
        return 0, 1

    def _split_list(self, items: list, n: int) -> list[list]:
        if n <= 1:
            return [items]
        chunk_size = int(math.ceil(len(items) / n))
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _inference_output_dir(self, model_id: str) -> str:
        return f"../inference/inference_results/{model_id}/{self.test_set_name}"

    def _answers_path(self, model_id: str, world_size: int, rank: int) -> str:
        return os.path.join(self._inference_output_dir(model_id), f"{world_size}_{rank}.jsonl")

    def _merge_path(self, model_id: str) -> str:
        return os.path.join(self._inference_output_dir(model_id), "merge.jsonl")

    def _run_inprocess_test_generation(self, model, model_id: str):
        """Generate merge.jsonl-compatible outputs using the already-loaded model on each rank."""
        processor = self.processor
        if processor is None:
            raise RuntimeError(
                "PeriodicCadIoUEvalCallback requires `processor` when use_inprocess_test=True"
            )

        rank, world_size = self._get_dist_info()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Read questions.
        with open(os.path.expanduser(self.question_file), "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f]

        chunks = self._split_list(questions, world_size)
        my_questions = chunks[rank] if rank < len(chunks) else []

        out_dir = self._inference_output_dir(model_id)
        os.makedirs(out_dir, exist_ok=True)
        answers_file = self._answers_path(model_id, world_size, rank)

        # Put model in eval mode for generation.
        was_training = getattr(model, "training", False)
        try:
            model.eval()
            with open(answers_file, "w", encoding="utf-8") as ans_f:
                for q in tqdm(my_questions, disable=not _is_rank_zero(), desc=f"eval_gen rank{rank}"):
                    image_file = q["image"]
                    qs = q["text"]
                    if os.path.isabs(image_file):
                        image_path = image_file
                    elif self.image_folder:
                        image_path = os.path.join(self.image_folder, image_file)
                    else:
                        image_path = image_file
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": qs},
                            ],
                        }
                    ]

                    inputs = processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    # Move tensors to device.
                    for k, v in list(inputs.items()):
                        if torch.is_tensor(v):
                            if k in ("pixel_values", "pixel_values_videos"):
                                inputs[k] = v.to(device, dtype=torch.bfloat16)
                            else:
                                inputs[k] = v.to(device)

                    input_ids = inputs["input_ids"]
                    attention_mask = inputs.get("attention_mask")
                    pixel_values = inputs.get("pixel_values")
                    image_grid_thw = inputs.get("image_grid_thw")
                    video_grid_thw = inputs.get("video_grid_thw")
                    pixel_values_videos = inputs.get("pixel_values_videos")

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            video_grid_thw=video_grid_thw,
                            pixel_values_videos=pixel_values_videos,
                            do_sample=True if (self.temperature and self.temperature > 0) else False,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            num_beams=self.num_beams,
                            max_new_tokens=self.max_new_tokens,
                            use_cache=True,
                        )

                    generated_ids = output_ids[:, input_ids.shape[1] :]
                    outputs = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    ans_id = uuid.uuid4().hex
                    ans_f.write(
                        json.dumps(
                            {
                                "question_id": q["question_id"],
                                "prompt": qs,
                                "text": outputs,
                                "answer_id": ans_id,
                                "model_id": model_id,
                                "metadata": {},
                            }
                        )
                        + "\n"
                    )
        finally:
            if was_training:
                model.train()

        _dist_barrier()

        # Merge on rank 0.
        if rank == 0:
            merge_path = self._merge_path(model_id)
            with open(merge_path, "w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    part_path = self._answers_path(model_id, world_size, r)
                    if not os.path.exists(part_path):
                        continue
                    with open(part_path, "r", encoding="utf-8") as in_f:
                        shutil.copyfileobj(in_f, out_f)

        _dist_barrier()

    def _get_writer(self):
        # Only rank 0 should write TensorBoard events to avoid log corruption.
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        if rank != 0:
            return None
        if self._writer is not None:
            return self._writer
        if SummaryWriter is None:
            return None
        log_dir = getattr(self.training_args, "logging_dir", None) or os.path.join(self.training_args.output_dir, "runs")
        self._writer = SummaryWriter(log_dir=log_dir)
        return self._writer

    def _checkpoint_dir(self, global_step: int) -> str:
        return os.path.join(self.training_args.output_dir, f"checkpoint-{global_step}")

    def _compute_eval_iou_from_merge(self, *, model_id: str) -> tuple[str, dict]:
        """Compute eval IoU directly from merge.jsonl using `step` paths in the eval JSONL.

        Returns (results_path, metrics_dict).
        """
        question_path = os.path.expanduser(self.question_file)
        with open(question_path, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]
        step_by_qid = {q.get("question_id"): q.get("step") for q in questions if q.get("question_id") is not None}

        merge_path = self._merge_path(model_id)
        out_dir = self._inference_output_dir(model_id)
        os.makedirs(out_dir, exist_ok=True)
        results_path = os.path.join(out_dir, "cad_iou_results.txt")

        if not os.path.exists(merge_path):
            Path(results_path).write_text("Average IoU: 0.0\nNumber of valid steps: 0\n", encoding="utf-8")
            return results_path, {"avg_iou": 0.0, "valid_steps": 0, "per_item": []}

        per_item = []
        ious: list[float] = []
        valid_steps = 0

        with open(merge_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ans = json.loads(line)
                qid = ans.get("question_id")
                raw_text = ans.get("text") or ""
                code = extract_final_code(raw_text)
                gt_step = step_by_qid.get(qid)

                if gt_step and os.path.exists(gt_step) and code:
                    iou = compute_iou_from_step(gt_step, code, timeout=IOU_TIMEOUT_SECONDS)
                    valid_steps += 1
                else:
                    iou = 0.0

                ious.append(float(iou))
                per_item.append({"question_id": qid, "iou": float(iou)})

        avg_iou = float(sum(ious) / len(ious)) if ious else 0.0

        # Write in the same format expected by `_parse_cad_iou_results`.
        lines = [
            f"Average IoU: {avg_iou}",
            f"Number of valid steps: {valid_steps}",
            "",
            "Per-sample IoU:",
        ]
        for row in per_item:
            lines.append(f"{row.get('question_id')}\t{row.get('iou')}")
        Path(results_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        return results_path, {"avg_iou": avg_iou, "valid_steps": valid_steps, "per_item": per_item}

    def on_save(self, args, state, control, **kwargs):
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
        except Exception:
            rank, world_size = 0, 1

        # Synchronize all ranks around evaluation.
        _dist_barrier()

        global_step = int(getattr(state, "global_step", 0) or 0)
        ckpt_dir = self._checkpoint_dir(global_step)
        # Some setups may save to output_dir directly; fall back if checkpoint dir doesn't exist.
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = self.training_args.output_dir

        # Model identifier used by existing inference folder layout.
        model_id = os.path.basename(ckpt_dir.rstrip("/"))

        # 1) Test generation to produce merge.jsonl
        #    - In-process mode: all ranks participate in generation and rank 0 merges.
        #    - External mode: only rank 0 runs the script.
        test_res = None
        if self.use_inprocess_test:
            model = kwargs.get("model")
            if model is None:
                trainer = kwargs.get("trainer")
                model = getattr(trainer, "model", None) if trainer is not None else None
            if model is None:
                raise RuntimeError("PeriodicCadIoUEvalCallback: missing model in on_save kwargs")
            self._run_inprocess_test_generation(model, model_id)
        else:
            if rank == 0:
                # Fallback: run external script (will reload model; may OOM if training model still resident).
                env = os.environ.copy()
                if self.limit_eval_to_gpu0:
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                test_cmd = f'bash ./scripts/test_gencadcode.sh "{ckpt_dir}" "{self.test_set_name}"'
                test_res = _run_bash(test_cmd, cwd=self.src_dir, env=env, timeout=self.eval_timeout_seconds)
            _dist_barrier()

        # 2-4) CAD artifact generation + IoU compute + TensorBoard logging should happen once.
        gen_res = None
        iou_res = None
        if rank == 0:
            # Compute IoU directly from merge.jsonl using absolute STEP paths in the eval JSONL.
            results_path, _metrics = self._compute_eval_iou_from_merge(model_id=model_id)
            metrics = _parse_cad_iou_results(results_path)

            writer = self._get_writer()
            if writer is not None:
                # Log raw file content for traceability.
                if metrics.get("raw_text"):
                    writer.add_text("eval/cad_iou_results", metrics["raw_text"], global_step=global_step)

                avg_iou = metrics.get("avg_iou")
                valid_steps = metrics.get("valid_steps")
                if isinstance(avg_iou, (float, int)):
                    writer.add_scalar("eval/avg_iou", float(avg_iou), global_step)
                if isinstance(valid_steps, int):
                    writer.add_scalar("eval/valid_steps", float(valid_steps), global_step)
                if isinstance(avg_iou, (float, int)) and isinstance(valid_steps, int):
                    writer.add_scalar("eval/avg_iou_times_valid_steps", float(avg_iou) * float(valid_steps), global_step)

                # Log command stderr/stdout (truncated) to help debugging failures.
                def _log_proc(tag: str, proc: subprocess.CompletedProcess | None):
                    if proc is None:
                        return
                    out = (getattr(proc, "stdout", "") or "")[-20000:]
                    err = (getattr(proc, "stderr", "") or "")[-20000:]
                    if out:
                        writer.add_text(f"eval/{tag}_stdout", out, global_step=global_step)
                    if err:
                        writer.add_text(f"eval/{tag}_stderr", err, global_step=global_step)

                _log_proc("test_gencadcode", test_res)
                writer.flush()

        # Ensure all ranks wait until evaluation/logging finished.
        _dist_barrier()
        return control


def _is_rank_zero() -> bool:
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    except ValueError:
        return True


def _sanitize_prompt_for_json(prompt):
    """Return a JSON-serializable prompt with images replaced by '<image>' tokens."""
    if isinstance(prompt, list):
        sanitized = []
        for msg in prompt:
            if isinstance(msg, dict):
                msg_copy = dict(msg)
                content = msg_copy.get("content")
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            new_content.append({"type": "image", "image": "<image>"})
                        else:
                            new_content.append(item)
                    msg_copy["content"] = new_content
                sanitized.append(msg_copy)
            else:
                sanitized.append(msg)
        return sanitized
    if isinstance(prompt, dict):
        prompt_copy = dict(prompt)
        return prompt_copy
    return prompt


def format_prompt_for_logging(prompt) -> str:
    """Render the prompt into a readable string that mirrors model input structure."""
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        lines = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                lines.append(f"[{role}]")
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            lines.append("<image>")
                        elif isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                lines.append(text)
                        else:
                            lines.append(str(item))
                elif isinstance(content, str):
                    lines.append(content)
                else:
                    lines.append(str(content))
            else:
                lines.append(str(msg))
        return "\n".join([ln for ln in lines if ln is not None])

    return str(prompt)


def extract_first_image_from_prompt(prompt):
    """Return the first PIL.Image from a chat-style prompt, or None if not found."""
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            return item.get("image")
    return None


def completion_to_text(completion) -> str:
    """Render model completion (raw) into a readable string."""
    def _flatten_content(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        pieces.append(part.get("text", ""))
                    elif "content" in part:
                        pieces.append(_flatten_content(part.get("content")))
                    else:
                        pieces.append(str(part))
                else:
                    pieces.append(str(part))
            return "\n".join(filter(None, pieces))
        if isinstance(content, dict):
            return _flatten_content(content.get("content"))
        return str(content)

    return _flatten_content(completion)


def _sanitize_for_path(value):
    if value is None:
        return "sample"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    sanitized = sanitized.strip("._-")
    return sanitized or "sample"


def setup_code_logger(base_output_dir: Path):
    global CODE_LOG_DIR, CODE_LOG_COUNTER
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    CODE_LOG_DIR = Path(base_output_dir) / "code_logs" / timestamp
    CODE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_LOG_COUNTER = itertools.count()


def log_code_pair(ground_truth: str, generated: str, sample_id: str, iou: float, step_index: int | None):
    if CODE_LOG_DIR is None:
        return
    if not _is_rank_zero():
        return

    if step_index is None:
        step_index = next(CODE_LOG_COUNTER)
    global LAST_LOGGED_STEP_CODEPAIR
    global LOGGED_CODEPAIR_IDS
    if step_index != LAST_LOGGED_STEP_CODEPAIR:
        LOGGED_CODEPAIR_IDS = set()
        LAST_LOGGED_STEP_CODEPAIR = step_index
    # Only log every N steps
    if step_index % LOG_EVERY_N_STEPS != 0:
        return
    
    safe_sample_id = _sanitize_for_path(sample_id)
    if safe_sample_id in LOGGED_CODEPAIR_IDS:
        return
    LOGGED_CODEPAIR_IDS.add(safe_sample_id)
    step_dir = CODE_LOG_DIR / f"step_{step_index:07d}_{safe_sample_id}"
    step_dir.mkdir(parents=True, exist_ok=True)

    (step_dir / "ground_truth.py").write_text(ground_truth or "", encoding="utf-8")
    (step_dir / "generated.py").write_text(generated or "", encoding="utf-8")

    metadata = {
        "sample_id": sample_id,
        "iou": float(iou),
    }
    (step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LAST_LOGGED_STEP_CODEPAIR = step_index


def log_prompt_and_generations(
    prompt,
    completions: list,
    sample_id: str,
    ious: list,
    step_index: int | None,
    num_prompts_total: int | None = None,
    num_completions_total: int | None = None,
    num_generations_expected: int | None = None,
    num_generations_used: int | None = None,
):
    """Log the prompt and all generated completions with their IoU scores."""
    if CODE_LOG_DIR is None:
        return
    if not _is_rank_zero():
        return
    
    if step_index is None:
        step_index = next(CODE_LOG_COUNTER)
    global LAST_LOGGED_STEP
    if step_index == LAST_LOGGED_STEP:
        return
    # Only log every N steps
    if step_index % LOG_EVERY_N_STEPS != 0:
        return
    
    safe_sample_id = _sanitize_for_path(sample_id)
    step_dir = CODE_LOG_DIR / f"step_{step_index:07d}_{safe_sample_id}_prompt_gens"
    step_dir.mkdir(parents=True, exist_ok=True)
    
    # Log the prompt (rendered and sanitized)
    prompt_text = format_prompt_for_logging(prompt).strip()
    (step_dir / "prompt.txt").write_text(prompt_text or "[empty prompt]", encoding="utf-8")
    try:
        sanitized_prompt = _sanitize_prompt_for_json(prompt)
        (step_dir / "prompt.json").write_text(json.dumps(sanitized_prompt, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort: ignore JSON serialization errors
        pass

    # Log the input image if present
    try:
        img = extract_first_image_from_prompt(prompt)
        if img is not None:
            # Save as PNG for consistent viewing
            img_path = step_dir / "input_image.png"
            img.save(img_path, format="PNG")
    except Exception:
        # Best-effort: ignore image save errors
        pass
    
    # Log each generation with its IoU
    for i, (completion, iou) in enumerate(zip(completions, ious)):
        gen_file = step_dir / f"generation_{i+1}_iou_{iou:.4f}.txt"
        completion_text = completion_to_text(completion).strip()
        gen_file.write_text(completion_text, encoding="utf-8")
    
    # Log summary metadata
    metadata = {
        "sample_id": sample_id,
        "step_index": step_index,
        "num_generations": len(completions),
        "num_prompts_total": num_prompts_total,
        "num_completions_total": num_completions_total,
        "num_generations_expected": num_generations_expected,
        "num_generations_used": num_generations_used,
        "ious": [float(iou) for iou in ious],
        "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
        "max_iou": float(max(ious)) if ious else 0.0,
        "min_iou": float(min(ious)) if ious else 0.0,
    }
    (step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LAST_LOGGED_STEP = step_index


def cad_iou_reward(completions, prompts=None, completion_ids=None, **kwargs):
    gt_codes = kwargs.get("gt_code") or []
    gt_step_paths = kwargs.get("gt_step_path") or kwargs.get("gt_step") or []
    base_len = len(gt_step_paths) or len(gt_codes) or len(completions)
    sample_ids = kwargs.get("sample_id") or [None] * base_len
    rewards = []

    trainer_state = kwargs.get("trainer_state")
    step_index = getattr(trainer_state, "global_step", None)

    # Optional distributed alignment check (off by default).
    global REWARD_DEBUG_HAS_RUN
    if (not REWARD_DEBUG_HAS_RUN) and _env_flag("GRPO_REWARD_DEBUG", "0"):
        try:
            _debug_check_grpo_grouping(
                sample_ids=kwargs.get("sample_id") or [],
                num_generations=NUM_GENERATIONS,
                step_index=step_index,
            )
        finally:
            # Run once to avoid adding overhead every step.
            REWARD_DEBUG_HAS_RUN = True

    # Prefer prompts passed by TRL (_calculate_rewards passes `prompts` explicitly)
    if prompts is None:
        prompts = kwargs.get("prompts")
    if prompts is None:
        prompts = kwargs.get("prompt")
    if prompts is None:
        prompts = [None] * base_len

    num_completions_total = len(completions)

    # trainer = kwargs.get("trainer")
    # num_generations_expected = getattr(getattr(trainer, "args", None), "num_generations", NUM_GENERATIONS)
    num_generations_expected = NUM_GENERATIONS

    # Group by sample_id to collect all generations for each prompt.
    groups = []
    groups_by_id = {}

    total_items = min(
        len(completions),
        len(sample_ids),
        len(prompts),
        (len(gt_step_paths) if gt_step_paths else len(gt_codes)),
    )
    for idx in range(total_items):
        completion = completions[idx]
        gt_code = gt_codes[idx] if (gt_codes and idx < len(gt_codes)) else ""
        gt_step_path = gt_step_paths[idx] if (gt_step_paths and idx < len(gt_step_paths)) else None
        sample_id = sample_ids[idx] if idx < len(sample_ids) else None
        prompt = prompts[idx] if idx < len(prompts) else None

        if sample_id not in groups_by_id:
            group = {
                "sample_id": sample_id,
                "prompt": prompt,
                "gt_code": gt_code,
                "gt_step_path": gt_step_path,
                "completions": [],
                "ious": [],
            }
            groups_by_id[sample_id] = group
            groups.append(group)

        group = groups_by_id[sample_id]

        gt_code_extracted = extract_final_code(group["gt_code"])
        code = extract_code_from_completion(completion)

        if not code:
            iou = 0.0
        else:
            try:
                if group.get("gt_step_path"):
                    iou = compute_iou_from_step(group["gt_step_path"], code, timeout=IOU_TIMEOUT_SECONDS)
                else:
                    iou = compute_iou_from_codes(gt_code_extracted, code, timeout=IOU_TIMEOUT_SECONDS)
            except Exception as e:
                print(f"Error computing IoU for sample {sample_id} idx {idx}: {e}")
                iou = 0.0

        rewards.append(iou)
        group["completions"].append(completion)
        group["ious"].append(iou)

        gen_idx = len(group["completions"])
        log_code_pair(
            gt_code_extracted or "",
            code or "",
            f"{sample_id}_gen{gen_idx}",
            iou,
            step_index,
        )

    num_prompts_total = len(groups)
    for group in groups:
        log_prompt_and_generations(
            group["prompt"],
            group["completions"],
            group["sample_id"],
            group["ious"],
            step_index,
            num_prompts_total=num_prompts_total,
            num_completions_total=num_completions_total,
            num_generations_expected=num_generations_expected,
            num_generations_used=len(group["completions"]),
        )

    return rewards


def extract_code_from_completion(completion):
    """Normalize GRPO completions into raw python code strings."""

    def _flatten_content(content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        pieces.append(part.get("text", ""))
                    elif "content" in part:
                        pieces.append(_flatten_content(part.get("content")))
                    else:
                        pieces.append(str(part))
                else:
                    pieces.append(str(part))
            return "\n".join(filter(None, pieces))
        if isinstance(content, dict):
            return _flatten_content(content.get("content"))
        return str(content)

    if isinstance(completion, list):
        # Conversational completions: list of messages
        flattened = []
        for message in completion:
            if isinstance(message, dict):
                flattened.append(_flatten_content(message.get("content")))
            else:
                flattened.append(_flatten_content(message))
        completion_text = "\n".join(filter(None, flattened))
    else:
        completion_text = _flatten_content(completion)

    completion_text = completion_text or ""
    return extract_final_code(completion_text)


def extract_final_code(text: str) -> str:
    """
    Extract the CADQuery python code from a model response.
    """

    if not isinstance(text, str):
        return ""

    code = text

    # Strip markdown code fences if present
    code = re.sub(r"```[a-zA-Z0-9_+-]*\n|```", "", code)
    return code.strip()


def compute_iou_from_codes(gt_code: str, gen_code: str, timeout: float = 30.0) -> float:
    """Compute IoU by executing both gt and generated CADQuery code."""
    import base64

    gt_code_b64 = base64.b64encode((gt_code or "").encode("utf-8")).decode("utf-8")
    gen_code_b64 = base64.b64encode((gen_code or "").encode("utf-8")).decode("utf-8")
    script = f"""
import cadquery as cq
import numpy as np
import json
import sys
import traceback
import base64

def cq_align_shapes(source, target):
    # Simplified version from compute_iou.py
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    if v_source <= 0:
        return 0.0

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0.0
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        
        try:
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            
            IOU = intersect.Volume() / union.Volume()
        except:
            IOU = 0.0
        
        if IOU > best_IOU:
            best_IOU = IOU
    return min(best_IOU, 1.0)

def generate_solid(code):
    try:
        exec_globals = {{}}
        exec(code, exec_globals)
        solid = exec_globals.get('solid')
        if solid is None:
            print('[cad_iou] No solid produced. Available keys:', list(exec_globals.keys()), file=sys.stderr)
            return None
        return solid
    except Exception as exc:
        print('[cad_iou] Error executing code:', exc, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None

gt_code = base64.b64decode('{gt_code_b64}').decode('utf-8')
gen_code = base64.b64decode('{gen_code_b64}').decode('utf-8')

gt_solid = generate_solid(gt_code)
gen_solid = generate_solid(gen_code)

if gt_solid is None or gen_solid is None:
    print('[cad_iou] Missing solids: gt_solid is None? ', gt_solid is None, ', gen_solid is None? ', gen_solid is None, file=sys.stderr)
    print(0.0)
else:
    gt_wp = gt_solid if hasattr(gt_solid, 'val') else cq.Workplane(gt_solid)
    gen_wp = gen_solid if hasattr(gen_solid, 'val') else cq.Workplane(gen_solid)
    iou = cq_align_shapes(gen_wp, gt_wp)
    print(iou)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        result = subprocess.run([
            'bash', '-c', 
            'source ../../miniconda3/etc/profile.d/conda.sh && conda activate cad_iou && python ' + script_path
        ], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            if result.stderr:
                print(f"[cad_iou] subprocess stderr: {result.stderr}")
            return 0.0
    except subprocess.TimeoutExpired:
        print(f"[cad_iou] subprocess timed out after {timeout} seconds", file=sys.stderr)
        return 0.0
    except:
        return 0.0
    finally:
        os.unlink(script_path)


def compute_iou_from_step(gt_step_path: str, gen_code: str, timeout: float = 30.0) -> float:
    """Compute IoU between generated CADQuery code and a ground-truth STEP file."""
    import base64

    gt_step_path = str(gt_step_path or "").strip()
    if not gt_step_path:
        return 0.0
    if not os.path.exists(gt_step_path):
        print(f"[cad_iou] missing gt STEP path: {gt_step_path}")
        return 0.0

    step_b64 = base64.b64encode(gt_step_path.encode("utf-8")).decode("utf-8")
    gen_code_b64 = base64.b64encode((gen_code or "").encode("utf-8")).decode("utf-8")

    script = f"""
import cadquery as cq
from cadquery import importers
import numpy as np
import sys
import traceback
import base64

def cq_align_shapes(source, target):
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    if v_source <= 0 or v_target <= 0:
        return 0.0

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0.0
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        try:
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            IOU = intersect.Volume() / union.Volume()
        except Exception:
            IOU = 0.0
        if IOU > best_IOU:
            best_IOU = IOU
    return min(best_IOU, 1.0)

def generate_solid(code):
    try:
        exec_globals = {{}}
        exec(code, exec_globals)
        solid = exec_globals.get('solid')
        if solid is None:
            return None
        return solid
    except Exception:
        return None

gt_step_path = base64.b64decode('{step_b64}').decode('utf-8')
gen_code = base64.b64decode('{gen_code_b64}').decode('utf-8')

try:
    gt_wp = importers.importStep(gt_step_path)
except Exception as exc:
    print('[cad_iou] Error importing STEP:', gt_step_path, exc, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(0.0)
    raise SystemExit(0)

gen_solid = generate_solid(gen_code)
if gen_solid is None:
    print(0.0)
    raise SystemExit(0)

gt_wp = gt_wp if hasattr(gt_wp, 'val') else cq.Workplane(gt_wp)
gen_wp = gen_solid if hasattr(gen_solid, 'val') else cq.Workplane(gen_solid)

try:
    print(cq_align_shapes(gen_wp, gt_wp))
except Exception:
    traceback.print_exc(file=sys.stderr)
    print(0.0)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                'bash',
                '-c',
                'source ../../miniconda3/etc/profile.d/conda.sh && conda activate cad_iou && python ' + script_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return float(result.stdout.strip() or 0.0)
        if result.stderr:
            print(f"[cad_iou] subprocess stderr: {result.stderr}")
        return 0.0
    except subprocess.TimeoutExpired:
        print(f"[cad_iou] subprocess timed out after {timeout} seconds", file=sys.stderr)
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass

# Dataset for GRPO
class GRPODataset(torch.utils.data.Dataset):
    def __init__(self, data_args):
        from qwenvl.data import data_list
        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        self.list_data_dict = []
        for data in dataset_list:
            annotations = json.load(open(data["annotation_path"], "r"))
            base_name = Path(data["annotation_path"]).stem
            for idx, ann in enumerate(annotations):
                ann["data_path"] = data["data_path"]
                # Add gt_code
                ann["gt_code"] = ann["conversations"][1]["value"]
                # f360rec annotations include an absolute ground-truth STEP file path under `step`.
                ann["gt_step_path"] = ann.get("step") or ann.get("gt_step_path")
                # Keep the user's text (without <image> placeholders) and add the image explicitly in __getitem__.
                ann["_grpo_user_text"] = (
                    ann["conversations"][0]["value"]
                    .replace("<image>\n", "")
                    .replace("<image>", "")
                    .strip()
                )
                ann["_grpo_sample_id"] = (
                    ann.get("id")
                    or ann.get("question_id")
                    or f"{base_name}_{idx:06d}"
                )
                self.list_data_dict.append(ann)
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, idx):
        from PIL import Image
        item = self.list_data_dict[idx]
        image_path = _resolve_image_path(item["data_path"], item["image"])
        image = Image.open(image_path).convert("RGB")

        gt_step_path = item.get("gt_step_path")
        if gt_step_path:
            try:
                gt_step_path = _resolve_step_path(item["data_path"], gt_step_path)
            except Exception:
                # Best-effort: keep the raw path (reward will treat missing paths as IoU=0).
                gt_step_path = item.get("gt_step_path")

        user_text = item.get("_grpo_user_text") or ""
        # Qwen-style multimodal chat prompt: explicit image + text.
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        return {
            "prompt": prompt,
            # Keep a parallel `images` field for compatibility with pipelines that
            # pass images separately from the chat-template prompt.
            "images": [image],
            "gt_code": item.get("gt_code", ""),
            "gt_step_path": gt_step_path,
            "sample_id": item.get("_grpo_sample_id", f"sample_{idx:06d}"),
        }

class VLGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        if data_collator is not None:
            self.data_collator = data_collator

    def create_optimizer(self):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for _, p in self.model.named_parameters()
                    if p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            }
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train_grpo():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    setup_code_logger(Path(training_args.output_dir))

    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    if isinstance(resume_from_checkpoint, str):
        resume_from_checkpoint = resume_from_checkpoint.strip() or None
    if resume_from_checkpoint is not None and not os.path.exists(resume_from_checkpoint):
        raise FileNotFoundError(
            f"Requested --resume_from_checkpoint does not exist: {resume_from_checkpoint}"
        )


    attn_implementation="flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    data_args.model_type = "qwen3vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    global REWARD_TOKENIZER
    REWARD_TOKENIZER = tokenizer


    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    dataset = GRPODataset(data_args)


    grpo_config = GRPOConfig(
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        max_completion_length = MAX_COMPLETION_TOKENS,
        num_train_epochs=training_args.num_train_epochs,
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        bf16=True,
        report_to="tensorboard",
        loss_type="dr_grpo",
        scale_rewards="none",
        mask_truncated_completions=True,
        disable_dropout=True,
        importance_sampling_level="sequence",
    )


    callbacks = []

    trainer = VLGRPOTrainer(
        model=model,
        reward_funcs=cad_iou_reward,
        args=grpo_config,
        train_dataset=dataset,
        data_collator=lambda batch: batch,
    )

    trainer.add_callback(
        PeriodicCadIoUEvalCallback(
            training_args=training_args,
            processor=processor,
            test_set_name="f360rec_test_data_subset100",
            src_dir="./",
            use_inprocess_test=True,
            limit_eval_to_gpu0=False,
            question_file="../inference/f360rec_test_data_subset100.jsonl",
            image_folder=None,
        )
    )
 

    if torch.distributed.get_rank() == 0:
        print("[GRPO] num_generations (trainer):", getattr(trainer, "num_generations", None))


    if resume_from_checkpoint and torch.distributed.get_rank() == 0:
        print(f"[GRPO] Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    train_grpo()