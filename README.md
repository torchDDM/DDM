# Diffusion Deformable Model (DDM)
The official source code of "diffusion deformable model (DDM)".

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
* You can download the data at [ACDC](https://acdc.creatis.insa-lyon.fr/description/databases.html).
* In our experiment, we resampled all MRI scans with a voxel spacing of 1.5 x 1.5 x 3.15 mm, and saved the data with .mat.
  * Examples of the data we used can be downloaded at [here](https://drive.google.com/drive/folders/1G0i9YI0qY3GXq4tUqFn6OeQKvMuRX69q?usp=sharing).
  * Copy downloaded directory of "ACDC_dataset" to './data'.
   * The data in the directory "data_ED_ES" have the 3D MR scans at the end diastolic and at the end systolic phases, and their corresponding segmentation labels.
   * The data in the directory "data_ED2ES" have the 4D MR scans from the end diastolic to the end systolic phases, which is used in the test stage.

Training
===============
* DDM_train.py which is handled by train.sh file.
* You can run the code through "sh train.sh" in terminal.
* All parameter settings we used are written in ./config/DDM_train.json file.

Evaluation
===============
* DDM_test.py which is handled by test.sh file.
* You can run the code by running "sh test.sh" in terminal.
* Chekcpoint of our experiment can be downloaded at [pretrained_model](https://drive.google.com/drive/folders/1fBTqdPXeSaFguXwu0bUOtfYHMTecemmL?usp=sharing).
  * For using our pretrained model, please make a new directory ./pretrained_model and save the model in the directory.
  * Results of our experiment
  
   | Method | PSNR (dB) | NMSE (x10^(-8)) | Dice | Time (sec) |
   |--------| ----------| ----------------| -----| -----------|
   | Initial | 28.058 (2.205) |0.790 (0.516) | 0.642 (0.188) | - |
   | Ours    | 30.725 (2.579) |0.466 (0.432) | 0.802 (0.109) | 0.456 |
