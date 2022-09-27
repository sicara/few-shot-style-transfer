# few-shot-style-transfer
Core research question: Can photorealistic style transfer make few-shot image classification tasks more robust to the choice of the support set?
## Setup
### Requirements
You will need :
- Python version 3.8.2: 
    -  `pyenv install 3.8.2`
- The packages in the [requirements.txt](requirements.txt) file: 
    - with pip do: `pip install -r requirements.txt`

### Dataset
We used the [Amazon Berkeley Objects (ABO) Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html). Here are the steps to install locally the dataset, needed to run the script of this repo:
- download [ABO_metadata](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar)
- download [ABO_downscaled_images](https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar)
- decompress the 2 .tar downloaded folders: 
    - on Ubuntu, one can use the following command line: `tar xvf folder_name.tar`
    - the ABO_metadata is now in a folder listings and the ABO_downscaled_images in a folder images. You must organize the data as follow:
    ```
    abo_dataset
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

## References
[1] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu,
Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen,
Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. Abo: Dataset
and benchmarks for real-world 3d object understanding. CVPR, 2022.
