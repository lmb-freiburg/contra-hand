# HanCo Dataset & Contrastive Representation Learning for Hand Shape Estimation 
Code in conjunction with the publication: *Contrastive Representation Learning for Hand Shape Estimation*.

This repository contains code for inference of both networks:
The one obtained from self-supervised contrastive pre-training and the network trained supervisedly for hand pose estimation.
Additionally, we provide examples how to work with the HanCo dataset and release the pytorch Dataset that was used during our pre-training experiments.
This dataset is an extension of the [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand) dataset.

Visit our [project page](https://lmb.informatik.uni-freiburg.de/projects/contra-hand/) for additional information.


# Requirements

### Python environment

    conda create -n contra-hand python=3.6
    conda activate contra-hand
    conda install -c pytorch pytorch=1.6.0 torchvision cudatoolkit=10.2
    conda install -c conda-forge -c fvcore fvcore transforms3d
    pip install pytorch3d transforms3d tqdm pytorch-lightning imgaug open3d matplotlib
    pip install git+https://github.com/hassony2/chumpy.git


### Hand Pose Dataset

You either need the [full HanCo dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html) or the small [tester data sample](https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_tester.zip) (recommended).

### Random Background Images

As the hand pose dataset contains green screen images, randomized backgrounds can be used. For our dataset we used 2195 images from Flickr. As these were not all licensed in a permissive manner, we provide a set of background images to use with the dataset.
These can be found [here](https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_rnd_backgrounds.zip).


### MANO model

Our supervised training code uses the MANO Hand model, which you need to aquire seperately due to licensing regulations: https://mano.is.tue.mpg.de

In order for our code to work fine copy *MANO_RIGHT.pkl* from the MANO website to *contra-hand/mano_models/MANO_RIGHT.pkl*.

We also build on to of the great PyTorch implementation of MANO provided by [Yana Hasson et al.](https://github.com/hassony2/manopth), which was modified by us and is already contained in this repository.


### Trained models

We release both the MoCo pretrained model and the shape estimation network that was derived from it.

In order to get the trained models download and unpack them locally:


    curl https://lmb.informatik.uni-freiburg.de/data/HanCo/contra-hand-ckpt.zip -o contra-hand-ckpt.zip & unzip contra-hand-ckpt.zip 


# Code

This repository contains scripts that facilitate using the HanCo dataset and building on the results from our publication.

### Show dataset

You will need to download the HanCo dataset (or at least the tester).
This script gives you some examples on how to work with the dataset.

    python show_dataset.py <Path-To-Your-Local-HanCo-Directory>


### Use our MoCo trained model


There is a simple script that calculates the cosine similarity score for two hard coded examples:

    python run_moco_fw.py


There is the script we used to create the respective figure in our paper.

    python run_moco_qualitative_embedding.py

### Self-Supervised Training with MoCo

We provide a torch data loader that can be used as a drop-in replacement for MoCo training.
The data loader can be found here `DatasetUnsupervisedMV.py`. It has boolean
options that control how the data is provided, these are `cross_bg`, `cross_camera`, and
`cross_time`. The `get_dataset` function also shows the pre-processing that we use, which is
slightly different from the standard MoCo pre-processing.

### Use our MANO prediction model

The following script allows to run inference on an example image:

    run_hand_shape_fw.py <Path-To-Your-Local-HanCo-Directory>


