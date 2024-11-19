################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

def plot_training_progress(logging_dict, test_acc):
    """
    Plots training and validation loss over epochs, and training, validation, and test accuracy.
    
    Args:
      logging_dict: A dictionary with keys 'train_loss', 'val_loss', 'train_accuracy', 
                    'val_accuracy'.
    """
    epochs = range(1, len(logging_dict['train_loss']) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # plot 1: training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, logging_dict['train_loss'], label="Training Loss", color="blue", marker='o')
    plt.plot(epochs, logging_dict['val_loss'], label="Validation Loss", color="red", marker='o')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # plot 2: training and validation accuracy with test accuracy line
    plt.subplot(2, 1, 2)
    plt.plot(epochs, logging_dict['train_accuracy'], label="Training Accuracy", color="blue", marker='o')
    plt.plot(epochs, logging_dict['val_accuracy'], label="Validation Accuracy", color="red", marker='o')
    
    plt.axhline(y=test_acc, color='green', linestyle='--', label=f"Test Accuracy ({test_acc:.2f})")
    
    plt.title("Training, Validation, and Test Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    predicted_classes = predictions.argmax(dim=1)
    correct = (predicted_classes == targets).sum().item()
    accuracy = correct / targets.size(0)

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():  # disable gradient computation for faster evaluation
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(model.device), y_batch.to(model.device)
            x_batch = x_batch.view(x_batch.size(0), -1)
            predictions = model(x_batch)
            correct += (predictions.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

    avg_accuracy = correct / total

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation.
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    val_accuracies = []
    best_val_accuracy = 0.0
    logging_dict = {'train_loss': [],
                    'train_accuracy': [],
                    'val_loss': [],
                    'val_accuracy': []}

    # initialize model, loss module, and optimizer
    model = MLP(n_inputs=32*32*3, n_hidden=hidden_dims, n_classes=10, use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training loop including SGD and validation
    for epoch in range(epochs):
        # train phase
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for x_batch, y_batch in cifar10_loader['train']:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.view(x_batch.size(0), -1) 
            
            optimizer.zero_grad()
            predictions = model(x_batch)

            batch_accuracy = accuracy(predictions, y_batch)
            epoch_accuracy += batch_accuracy

            loss = loss_module(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
    
        avg_epoch_loss = epoch_loss / len(cifar10_loader['train'])
        avg_epoch_accuracy = epoch_accuracy / len(cifar10_loader['train'])
        logging_dict['train_loss'].append(avg_epoch_loss)
        logging_dict['train_accuracy'].append(avg_epoch_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")
        
        # validation phase
        val_loss = 0.0
        for x_val_batch, y_val_batch in cifar10_loader['validation']:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            x_val_batch = x_val_batch.reshape(x_val_batch.shape[0], -1)
            predictions = model.forward(x_val_batch)
            val_loss += loss_module(predictions, y_val_batch).item()

        avg_val_loss = val_loss / len(cifar10_loader['validation'])
        logging_dict['val_loss'].append(avg_val_loss)
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(val_accuracy)
        
        # save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

    # test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])

    logging_dict["val_accuracy"] = val_accuracies
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print(f"Test accuracy: {test_accuracy:.4f}")
    plot_training_progress(logging_dict, test_accuracy)
