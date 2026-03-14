# For training and inference
conda create -n vlmtrl python=3.10 -y
source ../../miniconda3/bin/activate
conda activate vlmtrl
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# For generating CAD and computing IoU
conda deactivate
conda create -n cad_iou python=3.10 -y
source ../../miniconda3/bin/activate
conda activate cad_iou
conda install conda-forge::cadquery==2.5.2 -y
pip install scipy
conda install -y -c conda-forge nlopt
pip install trimesh
pip install plyfile
pip install pandas
pip install tqdm

# For generating orthographic drawings
conda deactivate
conda create -n pyocc python=3.10 -y
source ../../miniconda3/bin/activate
conda activate pyocc
conda install conda-forge::pythonocc-core -y
conda install conda-forge::p7zip -y
pip install -r requirements_pyocc.txt
