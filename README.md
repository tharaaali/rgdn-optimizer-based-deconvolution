# Image Deconvolution 

## sweet-project aims to improve visual acuity 
learn optimizer rgdn based on "Gong, D., Zhang, Z., Shi, Q., van den Hengel, A., Shen, C., & Zhang, Y. (2020). Learning deep gradient descent optimization for image deconvolution. IEEE transactions on neural networks and learning systems, 31(12), 5468-5482."
## Problem definition

Non blind deconvolution problem it's when we have blurred image and the bluer-kernel, we need to get the latent image x, while:
y = k * x + n

* y ∈ R^m : blurred image
* k ∈ R^l : corresponding blur kernel k
* x ∈ R^n : latent image
* '*' denotes the convolution operator
* n ∈ R^m : denotes an i.i.d. white Gaussian noise term with unknown standard deviation (i.e. noise level).
giving y and k, we need to estimate x
We'll call the result we get as Xhat

## Data

The dataset we're using till now: DATASET FROM HRTR: http://chaladze.com/l5/ //could be updated later
all files size: 256 * 256
there are 6000 files in train data, 2000 files for test data

## Model evaluation

The evaluation will be done using MSE and SSIM between (k * Xhat) and y
So, to evaluate our trained model: convoluted the recovered photo (output of our model) with the bluer-kernel and compare it with the input photo to the model (ground-truth * bluer-kernel) comparing methods to use :MSE, SSIM

## Requirements to Run

	1. first download the dataset (http://chaladze.com/l5/)
		need to make sure that the dataset is in the directory:
		#path for dataset folder:
		DATASET ="PATH" 
	
		specify the directory path, where the trained model will be saved:
		#path where to save trained models folder:
		SAVE_MODEL_DIR='PATH"
	
		in case the trained model is ready to test, the directory path to the model should be spicified 
		#path for already trained models folder:
		MODEL_DIR='PATH'
	
		initially all these directories are set right 
	
	2. get environment ready:
         
         first time when start server/set environmnt: it's needed to run the comands to install the requirment:
             uncomment the line: # !pip install -r requirements.txt
          
          or you can install the libraries	 
			# install necessary tools
			!pip install pydoe -q
			!pip install torchsummary -q
			!pip install torchmetrics -q
			!pip install piq -q
			!pip install torch==1.10 
	 If you are not working in jupyterHub/server of the laboratory, you may also need to install:
		PyYAML
		tqdm
		sklearn
		scikit-image
		numba

## To run the code:

    1. End-to-end-non-blind-deconvolution:
	in this file the code for training a model from the beginning !!

    2. evaluation2: 
	This file: to load an already trained model and evaluate it!


## info about data

Features:
        Some information about the data:
        We're generating dataset : blur kernel k, and n noise , and then y by ourself
        the bluer-kernel is psf generated from the other part of sweet project, in other words, k is the psf of an eye of individuals having refractive 
