import torch
from torch import nn
import os
import numpy as np
from Network import MNIST_Classifier
from DataLoader import custom_dataloader
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import math
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

import gc

"""This script defines the training, validation and testing process.
"""

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class MyModel(object):

    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.network = MNIST_Classifier(model_configs.embed_channels, model_configs.hidden_channels, model_configs.num_layers, model_configs.num_classes).to(model_configs.device)
      
    def train(self, x_train_wr, x_train_sp, y_train, training_configs, x_valid_wr = None, x_valid_sp = None, y_valid = None):

        print("<===================================================================== Training =====================================================================>")

        start = timeit.default_timer()

        train_dataloader = custom_dataloader(x_train_wr, x_train_sp, y_train, batch_size=training_configs.batch_size, train=True)
        val_dataloader = custom_dataloader(x_valid_wr, x_valid_sp, y_valid, batch_size=training_configs.batch_size, train=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.network.parameters(),
                                          lr = training_configs.learning_rate,  
                                          weight_decay = training_configs.weight_decay)
        self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps = 5, t_total=training_configs.num_epochs)
        
        train_loss_history = []
        val_loss_history = []
        train_accuracy_history = []
        val_accuracy_history = []
        learning_rate_history = []

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

            train_accuracy = sum(1 for x,y in zip(train_preds_all, train_labels_all) if x == y) / len(train_labels_all)
            valid_accuracy = sum(1 for x,y in zip(val_preds_all, val_labels_all) if x == y) / len(val_labels_all)

            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(valid_accuracy)
            learning_rate_history.append(self.scheduler.get_last_lr())

            print("-"*30)
            print(f"EPOCH {epoch}: Train Loss {train_loss:.4f}, Valid Loss {val_loss:.4f}")
            print(f"EPOCH {epoch}: Train Accuracy {train_accuracy:.4f}, Valid Accuracy {valid_accuracy:.4f}")
            print("-"*30)
            
            self.scheduler.step()
            self.plot_metrics(self.model_configs.result_dir, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history, learning_rate_history)

            if (epoch) % training_configs.save_interval == 0:
                self.save(epoch)

        stop = timeit.default_timer()
        print(f"Training Time: {stop - start : .2f}s")
        
        

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

        print(f"Test Overall Accuracy: {np.sum(np.array(test_preds_all) == np.array(test_labels_all))/len(test_labels_all):.2f}")
        result = classification_report(test_labels_all, test_preds_all, labels = list(range(10)))
        print(result)

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
        checkpoint_path = os.path.join(self.model_configs.save_dir, self.model_configs.checkpoint_name + '-%d.ckpt'%int(epoch))
        os.makedirs(self.model_configs.save_dir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def plot_metrics(self, result_dir, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history, learning_rate_history):
        
        # print(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Train Loss', color='blue')
        plt.plot(val_loss_history, label='Validation Loss', color='orange')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(result_dir, self.model_configs.checkpoint_name +'_loss_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(train_accuracy_history, label='Train Accuracy', color='blue')
        plt.plot(val_accuracy_history, label='Validation Accuracy', color='orange')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(result_dir, self.model_configs.checkpoint_name +'_accuracy_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(learning_rate_history,)
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)

        plt.savefig(os.path.join(result_dir, self.model_configs.checkpoint_name + '_learning_rate_plot.png'))
        plt.close()
