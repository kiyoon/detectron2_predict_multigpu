# Overview
Prodict object detection on big video dataset. It splits the job to multiple GPUs, but note that it does NOT use multiple GPUs to predict a single image.  
For example, if you have 4 GPUs, you have to open 4 terminals and run like this.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --frames-input-dir "/path/to/input/frames" --output "/path/to/output" --divide-job-count 4 --divide-job-index 0
CUDA_VISIBLE_DEVICES=1 python main.py --frames-input-dir "/path/to/input/frames" --output "/path/to/output" --divide-job-count 4 --divide-job-index 1
CUDA_VISIBLE_DEVICES=2 python main.py --frames-input-dir "/path/to/input/frames" --output "/path/to/output" --divide-job-count 4 --divide-job-index 2
CUDA_VISIBLE_DEVICES=3 python main.py --frames-input-dir "/path/to/input/frames" --output "/path/to/output" --divide-job-count 4 --divide-job-index 3
```

# Dependencies

Detectron2 == 0.1
torch == 1.3.1	(or 1.4.0?)
torchvision == 0.4.2

```bash
conda create -n detectron0.1 python=3.6
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit==10.1 -c pytorch
conda install -c conda-forge opencv


git clone https://github.com/facebookresearch/detectron2
mv detectron2 detectron2_0.1
cd detectron2_0.1
git checkout v0.1
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install -e .
```
