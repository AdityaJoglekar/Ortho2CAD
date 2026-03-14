import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, processor):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        image_path = os.path.join(self.image_folder, image_file)
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": qs}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        return inputs, line["question_id"], qs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    inputs_list, question_ids, prompts = zip(*batch)
    # Since batch_size=1, just return
    return inputs_list[0], question_ids[0], prompts[0]


# DataLoader
def create_data_loader(questions, image_folder, processor, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    if "qwen3" in args.model_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    elif "qwen2.5" in args.model_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")

    processor = AutoProcessor.from_pretrained(args.model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, processor)

    for inputs, idx, cur_prompt in tqdm(data_loader, total=len(questions)):
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to("cuda", dtype=torch.bfloat16)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to("cuda")
        video_grid_thw = inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to("cuda")
        pixel_values_videos = inputs.get("pixel_values_videos")
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to("cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                pixel_values_videos=pixel_values_videos,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        generated_ids = output_ids[:, input_ids.shape[1]:]
        outputs = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": os.path.basename(args.model_path),
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-3B-Instruct")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)