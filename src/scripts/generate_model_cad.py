import argparse
from utils_generate_model import *
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import re

ROOT_CHECKPOINT_DIR = "../inference/inference_results"

def wait_for_file(file_path, timeout=2, check_interval=0.1):
    """Waits for a file to exist for a limited time."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.isfile(file_path):
            return True
        time.sleep(check_interval)  # Small delay before rechecking
    return False  # Timed out waiting for file

def process_cad(args_tuple):
    """Function to write and run a single Python script."""
    code, id_, code_dir, stl_dir, step_dir, pc_dir_base, pc_reps, code_language = args_tuple
    
    if "```python" in code:
        code = re.sub(r"```[a-zA-Z]*\n|```", "", code)

    # Checks if the code can be run, without any modifications. Checking for syntax errors
    file_path = f"{code_dir}/{id_}.py"
    write_python_file(code, file_path)
    valid_code = run_python_script(file_path)
    
    valid_stl = False
    valid_pc = False
    if valid_code: # only move to stl generation if there is valid code
        # Adds code to generate stl, checks that STL is generated.
        if code_language == "pythonocc":
            code += f"\nwrite_stl_file(body, \"{stl_dir}/{id_}.stl\")"
            raise ValueError("Implement STEP generation")
        elif code_language == "cadquery":
            code += f"\nimport cadquery as cq\ncq.exporters.export(solid, \"{stl_dir}/{id_}.stl\")\n"
            code += f"\nimport cadquery as cq\ncq.exporters.export(solid, \"{step_dir}/{id_}.step\")\n"
        else:
            raise TypeError("CAD code language not supported!")
        write_python_file(code, f"{code_dir}/{id_}.py")
        valid_stl = run_python_script(f"{code_dir}/{id_}.py")
        if not wait_for_file(f"{stl_dir}/{id_}.stl"): # checks that the .stl was actually created, adds a little delay in case it's slow to save. #TODO implement this also for the python files generation?
            valid_stl = False
            
        # Generate point clouds
        if valid_stl:
            for i in range(pc_reps):
                try:
                    out_pc = convert_stl_to_point_cloud(f"{stl_dir}/{id_}.stl", f"{pc_dir_base}_{i}/{id_}.ply", 2000, seed=42+i)
                    if os.path.isfile(f"{pc_dir_base}_{i}/{id_}.ply"):
                        valid_pc = True # Only no errors and pc file exists should this be set to true
                except Exception as e:
                    print(f"{id_} failed point cloud generation")

    return valid_code, valid_stl, valid_pc, id_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split/test dataset used.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of split/test dataset.")
    parser.add_argument("--model_tested", type=str, required=True, help="Name of model.")
    parser.add_argument("--code_language", type=str, required=True, help="Name of code language, cadquery or pythonocc are currently supported")
    parser.add_argument("--pc_reps", type=int, default=0, help="Number of reps of point cloud generation.")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel using multiple CPUs.")

    args = parser.parse_args()
    
    model_name = args.model_tested.split("/")[-1]
    print(f"Model name: {model_name}")
    
    model_code, ids, q_token_count, gt_token_count, output_token_count = read_jsonl(ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/merge.jsonl", "text", "question_id", "question_token_count", "ground_truth_token_count", "output_token_count")
    
    # Set up for code generation
    code_dir = ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/model_code"
    os.makedirs(code_dir, exist_ok=True)
    
    # Set up for STL generation
    stl_dir = ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/model_stl"
    os.makedirs(stl_dir, exist_ok=True)
    
    # Set up for STEP generation
    step_dir = ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/model_step"
    os.makedirs(step_dir, exist_ok=True)
    
    # Set up for point cloud generation
    pc_dir_base = ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/model_point_cloud"
    for i in range(args.pc_reps):
        os.makedirs(pc_dir_base + f"_{i}", exist_ok=True)
    
    input_data = [(model_code[i], ids[i], code_dir, stl_dir, step_dir, pc_dir_base, args.pc_reps, args.code_language) for i in range(len(model_code))]
    
    if args.parallel:
        num_workers = min(8, cpu_count())  # Use 8 CPUs or the max available
        with Pool(num_workers) as pool:
            cad_results = list(tqdm(pool.imap_unordered(process_cad, input_data), 
                                total=len(input_data), 
                                desc="Processing CAD tasks"))
            
        valid_codes, valid_stls, valid_pcs, ids_out = zip(*cad_results)  # Separates tuples into respective results
            
    else:
        raise ValueError("Only implemented with parallelization at this time")
        # TODO: implement without parallelization
    
    # Store results
    df = pd.DataFrame({"q_ids": list(ids_out), "model_valid_code": list(valid_codes), "model_valid_stl": list(valid_stls), "model_valid_point_clouds": list(valid_pcs)})
    code_valid_rate = df["model_valid_code"].sum()/len(df)
    stl_valid_rate = df["model_valid_stl"].sum()/len(df)
    pc_valid_rate = df["model_valid_point_clouds"].sum()/len(df)
    
    # Write stats to .txt file
    with open(ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/cad_gen_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Valid code: {code_valid_rate}" + "\n")  # Write each value followed by a newline
        f.write(f"Valid stl: {stl_valid_rate}" + "\n")  # Write each value followed by a newline
        f.write(f"Valid point cloud: {pc_valid_rate}" + "\n")  # Write each value followed by a newline
    
    df.to_csv(ROOT_CHECKPOINT_DIR + f"/{model_name}/{args.dataset_name}/results.csv")