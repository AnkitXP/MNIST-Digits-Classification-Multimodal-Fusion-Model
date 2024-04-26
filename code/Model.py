import torch
from torch import nn
import os
import numpy as np
from Network import MNIST_Classifier
from DataLoader import custom_dataloader
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import gc

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.network = MNIST_Classifier().to(model_configs.device)
      
    def train(self, x_train_wr, x_train_sp, y_train, training_configs, x_valid_wr = None, x_valid_sp = None, y_valid = None):

        print("<===================================================================== Training =====================================================================>")

        start = timeit.default_timer()

        train_dataloader = custom_dataloader(x_train_wr, x_train_sp, y_train, batch_size=training_configs.batch_size, train=True)
        val_dataloader = custom_dataloader(x_valid_wr, x_valid_sp, y_valid, batch_size=training_configs.batch_size, train=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr = training_configs.learning_rate,  
                                          weight_decay = training_configs.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=training_configs.num_epochs, eta_min = 1e-5)
        
        train_loss_history = []
        val_loss_history = []

        for epoch in tqdm(range(1, training_configs.num_epochs + 1), position=0, leave=True):

            self.network.train()

            train_labels_all = []
            train_preds_all = []
            train_running_loss = 0

            for idx, (train_wr, train_sp, train_labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):

                current_train_wr = torch.tensor(train_wr, dtype=torch.float32).to(self.model_configs.device)
                current_train_sp = torch.tensor(train_sp, dtype=torch.float32).to(self.model_configs.device)
                current_train_labels = torch.tensor(train_labels, dtype=torch.int64).to(self.model_configs.device)

                self.optimizer.zero_grad()

                predictions = self.network(current_train_wr, current_train_sp)
                prediction_labels = torch.argmax(predictions, dim=1)

                loss = self.cross_entropy_loss(predictions, current_train_labels)

                train_labels_all.extend(train_labels.cpu().detach())
                train_preds_all.extend(prediction_labels.cpu().detach())

                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()
            
            train_loss = train_running_loss / (idx +1)
            train_loss_history.append(train_loss)

            self.network.eval()
            
            val_labels_all = []
            val_preds_all = []
            val_running_loss = 0

            with torch.no_grad():
                for idx, (validation_wr, validation_sp, validation_labels) in enumerate(tqdm(val_dataloader, position=0, leave=True)):

                    current_validation_wr = torch.tensor(validation_wr, dtype=torch.float32).to(self.model_configs.device)
                    current_validation_sp = torch.tensor(validation_sp, dtype=torch.float32).to(self.model_configs.device)
                    current_validation_labels = torch.tensor(validation_labels, dtype=torch.int64).to(self.model_configs.device)

                    val_predictions = self.network(current_validation_wr, current_validation_sp)
                    val_prediction_labels = torch.argmax(val_predictions, dim=1)

                    val_labels_all.extend(validation_labels.cpu().detach())
                    val_preds_all.extend(val_prediction_labels.cpu().detach())

                    loss = self.cross_entropy_loss(val_predictions, current_validation_labels)
                    val_running_loss += loss.item()
            val_loss = val_running_loss / (idx + 1)
            val_loss_history.append(val_loss)

            print("-"*30)
            print(f"EPOCH {epoch}: Train Loss {train_loss:.4f}, Valid Loss {val_loss:.4f}")
            print(f"EPOCH {epoch}: Train Accuracy {sum(1 for x,y in zip(train_preds_all, train_labels_all) if x == y) / len(train_labels_all):.4f}, Valid Accuracy {sum(1 for x,y in zip(val_preds_all, val_labels_all) if x == y) / len(val_labels_all):.4f}")
            print("-"*30)
            self.scheduler.step()

            if (epoch) % training_configs.save_interval == 0:
                self.save(epoch)

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start : .2f}s")
        self.plot_loss_graphs(self.configs.result_dir, train_loss_history, val_loss_history)

    def evaluate(self, x_test_wr, x_test_sp, y_test):

        print("<===================================================================== Testing =====================================================================>")

        test_dataloader = custom_dataloader(x_test_wr, x_test_sp, y_test, batch_size=128, train=False)
            
        test_labels_all = []
        test_preds_all = []

        with torch.no_grad():
            for idx, (test_wr, test_sp, test_labels) in enumerate(tqdm(test_dataloader, position=0, leave=True)):

                current_test_wr = torch.tensor(test_wr, dtype=torch.float32).to(self.model_configs.device)
                current_test_sp = torch.tensor(test_sp, dtype=torch.float32).to(self.model_configs.device)
                current_test_labels = torch.tensor(test_labels, dtype=torch.int64).to(self.model_configs.device)

                test_predictions = self.network(current_test_wr, current_test_sp)
                test_prediction_labels = torch.argmax(test_predictions, dim=1)

                test_labels_all.extend(test_labels.cpu().detach())
                test_preds_all.extend(test_prediction_labels.cpu().detach())

        print(f"Test Accuracy: {np.sum(np.array(test_preds_all) == np.array(test_labels_all))/len(test_labels_all):.2f}")

    def predict_prob(self, x_predict_wr, x_predict_sp):
        print("<===================================================================== Prediction =====================================================================>")

        predict_dataloader = custom_dataloader(x_predict_wr, x_predict_sp, x_predict_sp, batch_size=128, train=False)
        self.network.eval()
        
        predict_proba_final = []
        
        with torch.no_grad():
            for idx, (x_predict_wr, x_predict_sp, _) in enumerate(tqdm(predict_dataloader, position=0, leave=True)):
                current_predict_wr = torch.tensor(x_predict_wr, dtype=torch.float32).to(self.model_configs.device)
                current_predict_sp = torch.tensor(x_predict_sp, dtype=torch.float32).to(self.model_configs.device)
                probabilities = torch.argmax(self.network(current_predict_wr, current_predict_sp))
                predict_proba_final.extend(probabilities.cpu().detach())

        return np.stack(predict_proba_final, axis=0)

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs.save_dir, 'mnist-classifier.ckpt')
        os.makedirs(self.configs.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def plot_loss_graphs(self, result_dir, train_loss_history, val_loss_history):
        
        """
        Save the loss plot as an image.

        Args:
        - train_loss_history (list): List of training loss values
        - val_loss_history (list): List of validation loss values
        - configs: Configuration object containing the path to save the image

        Returns:
        - None
        """
        print(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(result_dir, 'loss_plot.png'))
        plt.close()