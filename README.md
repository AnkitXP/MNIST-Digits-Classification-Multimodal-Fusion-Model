# MNIST-Digits-Classification
To run the code, please create a conda environment using the below command:

>> conda env create -f environment.yml

To access the code, change the directory to code,

>> cd code

To train the model, use the following command,

>> python main.py --mode train

Once trained, checkpoints will be saved in saved_models folder, to predict using one of the saved models, use:

>> python main.py --mode predict --load <checkpoint_name>

Thanks!
