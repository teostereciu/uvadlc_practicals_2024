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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

import torch

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
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == targets)
    accuracy = correct_predictions / len(targets)

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

    total_accuracy = 0.0
    total_samples = 0

    for x_batch, y_batch in data_loader:
        # flatten
        x_batch = x_batch.reshape(x_batch.shape[0], -1)

        # forward pass
        predictions = model.forward(x_batch)
        
        # compute accuracy for the current batch
        batch_accuracy = accuracy(predictions, y_batch)

        # accumulate batch accuracy
        total_accuracy += batch_accuracy * len(y_batch)
        total_samples += len(y_batch)

    # compute the average accuracy over all samples
    avg_accuracy = total_accuracy / total_samples

    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # set-up logger
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model = None
    logging_dict = {'train_loss': [],
                    'train_accuracy': [],
                    'val_loss': [],
                    'val_accuracy': []}

    # initialize model and loss module
    model = MLP(n_inputs=32*32*3, n_hidden=hidden_dims, n_classes=10)
    loss_module = model.loss
    
    # training loop including validation
    for epoch in range(epochs):
        model.clear_cache()
        
        # training phase
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for x_batch, y_batch in cifar10_loader['train']:
            # flatten the input images 
            x_batch = x_batch.reshape(x_batch.shape[0], -1)

            # forward pass
            predictions = model.forward(x_batch)

            # compute loss
            loss = loss_module.forward(predictions, y_batch)
            epoch_loss += loss

            # compute accuracy for this batch
            batch_accuracy = accuracy(predictions, y_batch)
            epoch_accuracy += batch_accuracy

            # backward pass and weight update
            dout = loss_module.backward(predictions, y_batch) 
            model.backward(dout)

            # update weights
            for module in model.modules:
                if isinstance(module, LinearModule):
                    module.params['weight'] -= lr * module.grads['weight']
                    module.params['bias'] -= lr * module.grads['bias']

            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        logging_dict['train_loss'].append(avg_epoch_loss)
        logging_dict['train_accuracy'].append(avg_epoch_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")
        
        # validation phase
        val_loss = 0.0
        for x_val_batch, y_val_batch in cifar10_loader['validation']:
            x_val_batch = x_val_batch.reshape(x_val_batch.shape[0], -1)
            predictions = model.forward(x_val_batch)
            val_loss += loss_module.forward(predictions, y_val_batch)

        avg_val_loss = val_loss / len(cifar10_loader['validation'])
        logging_dict['val_loss'].append(avg_val_loss)  
        val_accuracy = evaluate_model(model, cifar10_loader['validation'])
        logging_dict['val_accuracy'].append(val_accuracy)

        # save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

    # test best model
    cifar10_loader['test']
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])

    model = best_model
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
