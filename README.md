# AMASS preprocess
In this repository, we provide the code for preprocessing and visulization of the Amass dataset. 

## Pre-requisites
1. Download ALL "SMPL-X Gender Specific" body information from the [official webpage of AMASS dataset](https://amass.is.tue.mpg.de).
2. Unzip all *.tar.bz2 files.
3. Prepare Conda.

## Installation
1. Prepare clean and new conda environment with Python3.7. Below the environment name is "amass".
```
conda create --name amass python=3.7
conda activate amass
```

2. Clone this Repository.
```
https://github.com/cotton-ahn/amass_preprocess
cd amass_preprocess
```

3. Install [Human_Body_Prior](https://github.com/nghorbani/human_body_prior)
```
https://github.com/nghorbani/human_body_prior
cd human_body_prior
pip install -r requirements.txt
```
- (Tested on 23.12.2021) If you do until here, you would get the error of `ERROR: No matching distribution found for torch==1.8.2`. 
- Check your CUDA environment and install PyTorch properly. Below example is for cuda 11.0. Refer [HERE](https://pytorch.org/get-started/previous-versions/).
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
- Modify requirements.txt. The pytorch version should be same as the one you installed
- And run `pip install -r requirements.txt` again in the folder of `human_body_prior`.
- If you get error regarding `gcc` and `boost`, try to install it.
- i.e, `sudo apt install g++-7` or `sudo apt-get install libboost-all-dev`...

4. Install PyRender
- Follow the description in [Link](https://pyrender.readthedocs.io/en/latest/install/index.html#osmesa)

5. Download Gender-Specific [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) models.
- Download SMPL with "removed head bun" for AMASS dataset.

