# BLIPSrecon
Code for reproducing results in the BLIPS paper:

Anish Lahiri, Guanhua Wang, Sai Ravishankar, Jeffrey A. Fessler, (2021).
"Blind Primed Supervised (BLIPS) Learning  for MR Image Reconstruction."
IEEE Transactions on Medical Imaging
http://doi.org/10.1109/TMI.2021.3093770;
[arXiv preprint arXiv:2104.05028.](https://arxiv.org/abs/2104.05028)

The code is made up of two components: 
* Blind dictionary learning (MATLAB version 2020+)
* Supervised learning (with PyTorch > 1.7.0).
* Additionally, we used [BART](https://mrirecon.github.io/bart/) to generate the dataset.

### Blind dictionary learning
* Run `batchSOUP_DLMRI_randmask.m` to reconstruct an image.
* The input data should include `I1` (ground truth), `S` (sensitivity map) and `Q1` (sampling masks).

### Supervised learning
The supervised learning approach adopted the training set generated from the fastMRI project.
* `Preprocessing.ipynb` provides an example of the data-preprocessing.
* `MODL_DLMRI_Knee_vd.sh` give an example of training the neural network.
* The file `requirements.txt` denotes related Python packages.
