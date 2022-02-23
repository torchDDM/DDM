# Diffusion Deformable Model (DDM)


Implementation
===============
A PyTorch implementation of deep-learning-based model.
* Requirements
  * OS : Ubuntu / Windows
  * Python 3.6
  * PyTorch 1.4.0

Dataset
===============
* In our experiment, we used 4D cardiac MR scans provided by the Automated Cardiac Diagnosis Challenge (ACDC). 
* You can download the data at this [link](https://acdc.creatis.insa-lyon.fr/description/databases.html).

Training
===============
* DDM_train.py which is handled by train.sh
* You can run the code through "sh train.sh" in terminal.
* All parameter settings we used are written in ./config/DDM_train.json

Evaluation
===============
* DDM_test.py which is handled by test.sh
* You can run the code by running "sh test.sh" in terminal.
* Chekcpoint of our experiment can be downloaded at [pretrained_model](https://drive.google.com/drive/folders/1fBTqdPXeSaFguXwu0bUOtfYHMTecemmL?usp=sharing).
  * For using our pretrained model, please make a new folder ./pretrained_model and save the model in the folder.
  *Results of our experiment*
   | Method | PSNR (dB) | NMSE (x10^(-8)) | Dice | Time (sec) |
   |--------| ----------| ----------------| -----| -----------|
   | Initial | 28.058 (2.205) |0.790 (0.516) | 0.642 (0.188) | - |
   | Ours    | 30.725 (2.579) |0.466 (0.432) | 0.802 (0.109) | 0.456 |
