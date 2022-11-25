# few-shot-style-transfer
Core research question: Can photorealistic style transfer make few-shot image classification tasks more robust to the choice of the support set?
## Setup
### Requirements
You will need :
- Python version 3.8.2: 
    - `pyenv install 3.8.2`
    - Create a virtualenv with this version:
        ```bash
        pyenv virtualenv 3.8.2 name 
        pyenv local name
        ```
- The packages in the [requirements.txt](requirements.txt) file: 
    - with pip do: `pip install -r requirements.txt`
- CuPy, compatible with your cuda toolkit version:
    - check [this](https://docs.cupy.dev/en/stable/install.html) to know how to install CuPy adapted to your cuda toolkit version
    - for example, if `nvcc --version` says that you have the 10.1 release of the cuda toolkit, install CuPy with pip using `pip install cupy-cuda101`

### Dataset
We used the [Amazon Berkeley Objects (ABO) Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html). Here are the steps to install locally the dataset, needed to run the script of this repo:
- download ABO's metadata and images: `curl link.tar -L -O -J`
    - [ABO_metadata](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar)
    - [ABO_downscaled_images](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar)
- decompress the 2 .tar downloaded folders: 
    - on Ubuntu, one can use the following command line: `tar xvf folder_name.tar`
    - the ABO_metadata is now in a folder listings and the ABO_downscaled_images in a folder images. You must organize the data as follow:
    ```
    data
      |--abo_dataset
            |--listings
            |   |--metadata
            |       |--listings_*.json.gz
            |
            |--images
                |--metadata
                |    |--images.csv.gz
                |
                |--small
                     |--** 
                        |--********.jpg
    ```
    You may find useful:
    ```bash
    mkdir new_folder_name # create a folder (where the command is run)
    mv source target # move source folder into target folder
    ```
### Troubleshooting
- error while installing requirements? try: `pip install --upgrade pip`
- Difficulties installing pyenv?
    ```bash
    sudo apt-get install curl
    curl https://pyenv.run | bash
    ``` 
    in your zshrc `nano ~/.zshrc` add:
    ```bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    eval "$(pyenv init --path)"
    ```
- you don't have nvcc?  use: `sudo apt install nvidia-cuda-toolkit`
- you have problem using .cuda() with torch? install the torch version that correspond to your cuda version, the same that you used for cupy
- `"/usr/include/math_functions.h"` file is not found ? It may be in a different location for you. In `src/style_transfer/smooth_filter.py`, modify line 6 path with your own path
## Run inference script
`python -m scripts.main`
```bash
Options:
 --number-of-tasks INTEGER            [default: 100]
 --color-aware / --no-color-aware     [default: no-color-aware]
 --style-transfer-augmentation / --no-style-transfer-augmentation
                                      [default: no-style-transfer-augmentation]
 --save-results / --no-save-results   [default: save-results]
 --help                          Show help message and exit.

```
:warning: in order to save the results, you need a exp_results folder at the root of the repo: `mkdir exp_results`
## References
[1] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. CVPR, 2022.

[2] Yijun Li, Ming-Yu Liu, Xueting Li, Ming-Hsuan Yang, Jan Kautz. A Closed-form Solution to Photorealistic Image Stylization. CoRR, 2018.

[3] Bennequin, E. [easyfsl](https://github.com/sicara/easy-few-shot-learning) [Computer software].
