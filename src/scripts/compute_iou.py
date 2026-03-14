import cadquery as cq
import numpy as np
import os
import json
import argparse
from typing import Tuple, Union
import shutil
from tqdm import tqdm
import subprocess

def cq_align_shapes(source : cq.Workplane, target : cq.Workplane) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    if v_source <= 0:
        return None, 0.0, c_source, c_target
    else:

        s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
        s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

        normalized_source = source.translate(-c_source).val().scale(1/s_source)
        normalized_target = target.translate(-c_target).val().scale(1/s_target)

        Rs = np.zeros((4,3,3))
        Rs[0] = I_v_target @ I_v_source.T

        for i in range(3):
            # all possible 2 out of 3 permutations
            alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
            Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

        best_IOU = 0.0
        best_T = None
        for i in range(4):
            T = np.zeros([4,4])
            T[:3,:3] = Rs[i]
            T[-1,-1] = 1
            
            aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
            
            try:
                intersect = aligned_source.intersect(normalized_target)
                union = aligned_source.fuse(normalized_target)
                
                IOU = intersect.Volume() / union.Volume()
            except: #handle cases where IOU is undefined
                IOU = 0.0
            
            if IOU > best_IOU:
                best_IOU = IOU
                best_T = T
        if best_IOU > 1.0:
            print("Error: IOU greater than 1.0", best_IOU)
            best_IOU = 1.0
        if best_IOU < 0.2:
            print("Warning: Low IOU found:", best_IOU)
        if best_T is not None:
            aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
            print('IoU:', best_IOU)
            return cq.Workplane(aligned_source), best_IOU, c_source, c_target
        else:
            aligned_source = None
            print('IoU:', best_IOU)
            return aligned_source, best_IOU, c_source, c_target

def find_image_by_question_id(jsonl_path, target_question_id):
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)  # Parse JSON line
            if data.get("question_id") == target_question_id:
                return data.get("image")[:-4]  # Return the "image" field
    return None  # Return None if not found

def average_non_none(values):
    filtered_values = [v for v in values if v is not None]
    print(f"Number of Nones: {len(values) - len(filtered_values)}")
    return sum(filtered_values) / len(filtered_values) if filtered_values else None


# Write a main function that is called inside __main__
def main(model_path, test_set_name, ground_truth_steps_dir):
    model_name = model_path.split("/")[-1]
    model_generated_steps_dir = f"../inference/inference_results/{model_name}/{test_set_name}/model_step/"
    test_jsonl = f"../inference/{test_set_name}.jsonl"
    all_ious = []
    qid_iou_dict = {}
    gt_steps = []
    model_steps = []
    model_steps_aligned = []
    # Add tqdm to below
    for g in tqdm(os.listdir(model_generated_steps_dir)):
        print(f"Processing: {g}")
        question_id = g[:-5]
        if 'cadquery' in test_set_name:
            orig_id = find_image_by_question_id(test_jsonl, int(question_id))
        else:
            orig_id = find_image_by_question_id(test_jsonl, question_id)
        if orig_id == None:
            raise ValueError("Can't find original ID in test set")
        
        gt_step = cq.importers.importStep(ground_truth_steps_dir + orig_id + ".step")
        model_generated_step = cq.importers.importStep(model_generated_steps_dir + g)
        gt_steps.append(gt_step)
        model_steps.append(model_generated_step)
        try:
            aligned_model_generated, IOU, _, _ = cq_align_shapes(model_generated_step, gt_step)
            model_steps_aligned.append(aligned_model_generated)
            all_ious.append(IOU)
            qid_iou_dict[question_id] = (orig_id, IOU)

        except: #fix:only gemini seemed to have this problem, added try except statement for gemini
            print("issue")

    # Create histogram of IoU scores using matplotlib in conda pyocc environment
    hist_command = f"""
source ../../miniconda3/etc/profile.d/conda.sh && 
conda activate pyocc && 
python3 -c "
import matplotlib.pyplot as plt
import json
ious = {all_ious}
plt.figure(figsize=(10,6))
plt.hist(ious, bins=100, edgecolor='black')
plt.title('Distribution of IoU Scores')
plt.xlabel('IoU Score')
plt.ylabel('Count')
plt.savefig('../inference/inference_results/{model_name}/{test_set_name}/iou_histogram.png')
plt.close()"
"""
    subprocess.run(hist_command, shell=True, executable='/bin/bash')

    print(f"Model's average IoU score: {average_non_none(all_ious)}")
    with open(f"../inference/inference_results/{model_name}/{test_set_name}/qid_iou_dict.json", "w", encoding="utf-8") as f:
        json.dump(qid_iou_dict, f, indent=4)
    
    iou_result_file = f"../inference/inference_results/{model_name}/{test_set_name}/cad_iou_results.txt"
    with open(iou_result_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test set: {test_set_name}\n")
        f.write(f"Average IoU: {average_non_none(all_ious)}\n")
        f.write(f"Number of valid steps: {len(all_ious)}\n")
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute model's IoU score.")
    parser.add_argument("--model_path", type=str, required=True, help="Model to compute IoU for.")
    parser.add_argument("--test_set_name", type=str, required=True, help="Name of the test set.")
    parser.add_argument("--ground_truth_steps_dir", type=str, default="../inference/test100_gt_steps/", help="Directory containing ground truth STEP files.")
    args = parser.parse_args()

    main(args.model_path, args.test_set_name, args.ground_truth_steps_dir)