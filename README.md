# Skull-to-Face: Anatomy-Guided 3D Facial Reconstruction and Editing

Project page: https://xmlyqing00.github.io/skull-to-face-page/


## Environment

```bash
conda create -n skull2face python=3.9
conda activate skull2face

# Please install the correct version of pytorch and torchvision
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c iopath iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install other dependencies
pip install -r requirements.txt
```

## Data
Download the data from FLAME official website: https://flame.is.tue.mpg.de/, please require a proper license to use the data.
For academic testing purpose, I compressed the necessary pretrained models in the Google Drive.
Download it and unzip the four files in the `assets` folder.

## Run the code

```bash
 python fit_skull_real.py --cfg configs/fit_skull_real_robert.yaml 
```
The results will be saved in the `output` folder.

