# Multimodal matching using a Hybrid Convolutional Neural Network
# \t\tFinal Project by 304977861 & 317179000

# Description
This is our implemetation for the paper "Multimodal matching using a Hybrid Convolutional Neural Network" by Elad Ben Baruch and Prof. Yosi Keller.
This is project is part of Tel Aviv University Deep Learning course.

# Prerequisite
  1. Python 3.6
  2. Install interperter dependencies from the requirements.txt
     Mainly: Tensorflow 2.0, numpy 1.18.1, opencv 3.4.2, pandas 0.25.3, matplotlib 3.1.1
  3. VEDAI dataset   
  4. Trained models (h5 files)

# VEDAI Dataset
  download the original dataset from the link (only the 512x512 images - two parts) - 
  https://downloads.greyc.fr/vedai/
  
  Extarct the rar files in "Datasets\VEDAI\Vehicules512"
  Copy the following excel files to "VEDAI" directory
    • vedaiTrain.csv
    • vedaiTest.csv
    • vedaiTestDetections.csv

# Code
  Download the "Code/main.py" and the trained model "Code/hybridModel_Epoch40_vedai.h5"
  Train, evaluation or vizualization can be choose in the main function configuration.
  Make sure to set the base path correctly.
    
