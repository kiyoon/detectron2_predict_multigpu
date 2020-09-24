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
