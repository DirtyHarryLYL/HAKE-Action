# Installation

### Requirements

- Ubuntu 16.04
- Nvidia GPU
- Cuda 8.0, cudnn 5.1
- Protobuf 2.6.1
- Numpy
- Pillow
- scikit-image
- scipy
- six
- wheel
- OpenCV

## Instructions
#### 1. Clone the code

```
git clone -b Image-level-HAKE-Action //github.com/DirtyHarryLYL/HAKE-Action.git
cd HAKE-Action
```

#### 2. Install the requirements
We tested the code on Ubuntu 16.04 and cuda 8.0/cudnn 5.1.

- Build a conda environment for caffe.

```
conda create -n caffe python=2.7
source activate caffe
```

- Install the requirements.

```
pip install -r requirements.txt
```

- Add the following lines to your bashrc

```
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-8.0
export PYTHONPATH=xxx/HAKE-Action/python:$PYTHONPATH
```
then ```source ~/.bashrc```.

#### 3. Build caffe

```
make all -j8
make pycaffe
```

#### 4. Test

```
python -c "import caffe"
```


You can refer to [this website](https://caffe.berkeleyvision.org/install_apt.html) for detailed instructions.


#### 4. (Optional) Build from Docker(TODO)


