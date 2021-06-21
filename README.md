# BLIPSrecon
Code for reproducing results in the BLIPS paper:

[Lahiri, A., Wang, G., Ravishankar, S., & Fessler, J. A. (2021). Blind Primed Supervised (BLIPS) Learning  for MR Image Reconstruction. *arXiv preprint arXiv:2104.05028*.](https://arxiv.org/abs/2104.05028)

The code is made up of two components: 

Blind dictionary learning (on MATLAB) and Supervised learning (with PyTorch > 1.7.0).
Additionally, we used [BART](https://mrirecon.github.io/bart/) for generating the dataset. 

The MATLAB code runs well on MATLAB 2020+.
Run `batchSOUP_DLMRI_randmask.m` to reconstruct an image.
The input data should include `I1` (ground truth), `S` (sensitivity map) and `Q1` (sampling masks).

* The training set is generated from the fastMRI project.
* The notebook `Preprocessing.ipynb` provides an example of the data-preprocessing.
* The script `Training.sh` gives an example of training the neural network.
* The file `requirements.txt` denotes related Python packages.



