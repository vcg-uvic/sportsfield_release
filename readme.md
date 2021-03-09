# Optimization Based Image Registration

## Optimizing Through Learned Errors for Accurate Sports Field Registration - WACV 2020
This repository is a reference implementation for the inference part of "Optimizing Through Learned Errors for Accurate Sports Field Registration", WACV 2020. 
For more details, please refer to our WACV 2020 or [[arXiv](https://arxiv.org/abs/1909.08034)] paper. A video showing the results is available [[here](https://jiangwei221.github.io/vids/sportsfield/README.html)]

![teaser](https://raw.githubusercontent.com/vcg-uvic/sportsfield/master/data/teaser.png)

The released code is freely available for free non-commercial academic research use, and may be redistributed under these conditions. Any commercial use is prohibited.

*Note*: We decided to not release the training code. Sorry for any inconvenience.

## Patent Pending
The Optimization Based Image Registration is patent protected (pending applications **US 62/850,910**; **US 16/049,546**; **EP17746676.0**; **CA 3,012,721**)[[here](https://patents.google.com/patent/US20200372679A1/en)] and shall not be used for any commercial application. For information about licensing please contact If you are interested in a commercial license, contact [Sportlogiq](https://sportlogiq.com) or [SLiQ Labs](https://sliqlabs.com). 

## Content of the repository
1. The trained weights (both initial guess net, and loss surface net) for soccer.
2. The inference code for soccer.
3. A Jupiter notebook to for simple user interaction.
4. The code to generate a soccer field template(Processing language) and a h5 format test dataset used in the paper.

### Installation

This implementation is based on Python3 and PyTorch.

You can install the environment by: ```conda env create -f environment.yml```

Activate the env by: ```conda activate sportsfield```

### Pretrained Weights

We provide the pretrained weights for soccer on [Google drive](https://drive.google.com/uc?id=1kgc6wfgdIDsHBhFMAr6YwTWbrigNv_UB&export=download). Download "out.zip", and extract all the content to  ```./out```, such that the ```./out``` folder contains ```pretrained_init_guess``` and ```pretrained_loss_surface``` .

### Play with jupyter notebook

Users can overlay the template to a soccer image or video using the notebook.

### Evaluation

Users can simply run: `python test_end2end.py loss_surface init_guess --load_weights_upstream "pretrained_init_guess" --load_weights_error_model "pretrained_loss_surface" --batch_size 32` to start the evaluation.

A reference evaluation result is provided for comparison:
```
----- Summary -----
original IOU part mean: 0.90211654
original IOU part median: 0.91872334
original IOU whole mean: 0.8406853
original IOU whole median: 0.857767
optimized IOU part mean: 0.9530167
optimized IOU part median: 0.9701195
optimized IOU whole mean: 0.9019278
optimized IOU whole median: 0.9253305
----- -----
spent 290.74491572380066 seconds for 186 images
1.5631447081924768 seconds per single image
----- End -----
```

## Citation
If you use this code in your research, cite the paper: 

```
@inproceedings{jiang2020optimizing,
author={Wei Jiang and Juan Camilo Gamboa Higuera and Baptiste Angles and Weiwei Sun and Mehrsan Javan and Kwang Moo Yi},
booktitle={2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},
title={Optimizing Through Learned Errors for Accurate Sports Field Registration},
year={2020},
organization={IEEE}
}
```

## License
The released code is freely available for free non-commercial academic research use, and may be redistributed under these conditions. Please, see the [license](LICENSE) for further details. If you are interested in a commercial license, contact [Sportlogiq](https://sportlogiq.com) or [SLiQ Labs](https://sliqlabs.com) for licensing information. 

**Note**: The Optimization Based Image Registration is patent protected (pending applications *US 62/850,910*; *US 16/049,546*; *EP17746676.0*; *CA 3,012,721*) and shall not be used for any commercial application. 
