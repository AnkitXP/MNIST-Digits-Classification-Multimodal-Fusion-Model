import os, argparse
import numpy as np
import pandas as pd
from Model import MyModel
from DataLoader import load_train_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
import torch
import random

import warnings
warnings.filterwarnings('ignore')

def configure():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, help="train, test or predict", required=True)
	parser.add_argument("--load", type=str, help="model checkpoint load path", default='mnist-classifier-with-augmentation-10.ckpt')
	return parser.parse_args()

def main(args, model_configs):

	torch.manual_seed(model_configs.random_seed)
	np.random.seed(model_configs.random_seed)
	random.seed(model_configs.random_seed)

	model = MyModel(model_configs)

	if args.mode == 'train':
		
		# checkpoint_path = os.path.join(model_configs.save_dir, args.load) 
		# model.load(checkpoint_path)

		x_train_wr, x_train_sp, y_train = load_train_data(model_configs.data_dir)
		x_train_wr, x_train_sp, y_train, x_test_wr, x_test_sp, y_test = train_valid_split(x_train_wr, x_train_sp, y_train, val_ratio=model_configs.test_ratio)
		x_train_wr, x_train_sp, y_train, x_valid_wr, x_valid_sp, y_valid = train_valid_split(x_train_wr, x_train_sp, y_train, val_ratio=model_configs.val_ratio)

		model.train(x_train_wr, x_train_sp, y_train, training_configs, x_valid_wr, x_valid_sp, y_valid)
		model.evaluate(x_test_wr, x_test_sp, y_test)

	elif args.mode == 'predict':

		checkpoint_path = os.path.join(model_configs.save_dir, args.load) 
		model.load(checkpoint_path)

		x_test_wr, x_test_sp = load_testing_images(model_configs.data_dir)
 
		predictions = model.predict_prob(x_test_wr, x_test_sp)
		predictions_df = pd.DataFrame({'row_id': range(len(predictions)), 'label': predictions})
		predictions_df.to_csv(model_configs.result_dir + 'predictions.csv', index=False)

if __name__ == '__main__':
	
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	args = configure()
		
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model_configs.device = device
	
	main(args, model_configs)
