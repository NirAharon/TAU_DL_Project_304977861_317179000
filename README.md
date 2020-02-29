# Final Project by 304977861 & 317179000
## Multimodal matching using a Hybrid Convolutional Neural Network

### Description
Our implemetation of the paper "Multimodal matching using a Hybrid Convolutional Neural Network" by Elad Ben Baruch and Prof. Yosi Keller.

This project is part of Tel Aviv University Deep Learning course 0510-7255.

### Prerequisite
  1. Python 3.6
  2. Install interperter dependencies from the "Code\requirements.txt". Mainly:
  
    • Tensorflow 2.0
     
    • numpy 1.18.1
     
    • opencv 3.4.2
     
    • pandas 0.25.3
     
    • matplotlib 3.1.1
     
  3. VEDAI dataset   
  4. Trained models (.h5 file)

### VEDAI Dataset
  Download the original dataset from the [link](https://downloads.greyc.fr/vedai/) (only the 512x512 images - two parts).
  
  In case the original link is broken, the dataset was downloaded and uploaded to google drive [link](https://downloads.greyc.fr/vedai/)
    
  Extarct the .rar files to directory "Datasets\VEDAI\Vehicules512"
  
  Copy the following excel files to "VEDAI" directory
    
    • vedaiTrain.csv
    
    • vedaiTest.csv
    
    • vedaiTestDetections.csv

### Code
  Download the "Code/main.py" and the trained model "Code/hybridModel_Epoch40_vedai.h5"
  
  Train, evaluation or vizualization can be chosen in the main function configuration.
  
  Make sure to set the "basePath" in main.py correctly.
    
