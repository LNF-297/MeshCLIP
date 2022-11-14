# MeshCLIP: Efficient and Robust  3D Mesh Understanding with Zero/Few-shot Approach

Official implementation of "MeshCLIP: Efficient and Robust  3D Mesh Understanding with Zero/Few-shot Approach".


## Requirements

### Environment

cuda: >=11.6

pytorch: >=1.11.0

python: >=3.7 

pytorch3d: >=[0.7.0](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.7.0)

### Installation

Create a conda environment with **python 3.7.12**.

```bash
conda create -n meshclip python=3.7
conda activate meshclip
```



Install pytorch that fits your cuda version.

```bash
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```



Install pytorch3d

Refer to [pytorch3d/INSTALL.md at main · facebookresearch/pytorch3d (github.com)](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

**Notice:** The version of pytorch3d used in our code should be built from the source by 

`pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`.



Install other dependences:

```bash
pip install -r requirements.txt
```



### Dataset Preparation

+ Download and extract original datasets to `./data`.

  ```
  |-- data
  	|-- 3D-FUTURE-model
  	|-- Manifold40
  	|-- ModelNet40
  ...
  ```

+ Run the tool scripts in `./prepare_data` to create index and normalize the data.

  **Notice:** To run the script, "Meshlab server" are required.

+ There might be some segmentation errors in original MoldelNet40 dataset, use `./prepare_data/modelnet40_fix.py` to fix it before running scripts. 

## Get started

### Zero-shot

#### ModelNet40

```bash
python main_cls.py --trainer zeroshot --dataset_name ModelNet40 --dataset_path ./data/ModelNet40_Processed --mesh_views=10
```

#### Manifold40

```bash
python main_cls.py --trainer zeroshot --dataset_name Manifold40 --dataset_path ./data/Manifold40_Processed --mesh_views=10
```

#### 3D-FUTURE (sub categories)

```bash
python main_cls.py --trainer zeroshot --dataset_name 3D-FUTURE_sub --dataset_path ./data/3D-FUTURE-model --mesh_views=8
```

#### 3D-FUTURE (super categories)

```bash
python main_cls.py --trainer zeroshot --dataset_name 3D-FUTURE_super --dataset_path ./data/3D-FUTURE-model --mesh_views=8
```

### Few-shot

#### ModelNet40

```bash
python main_cls.py --trainer fewshot --dataset_name ModelNet40 --dataset_path ./data/ModelNet40_Processed --mesh_views=14
```

#### Manifold40

```bash
python main_cls.py --trainer fewshot --dataset_name Manifold40 --dataset_path ./data/Manifold40_Processed --mesh_views=14
```

#### 3D-FUTURE (sub categories)

```bash
python main_cls.py --trainer fewshot --dataset_name 3D-FUTURE_sub --dataset_path ./data/3D-FUTURE-model --mesh_views=14
```

#### 3D-FUTURE (super categories)

```bash
python main_cls.py --trainer fewshot --dataset_name 3D-FUTURE_super --dataset_path ./data/3D-FUTURE-model --mesh_views=14
```