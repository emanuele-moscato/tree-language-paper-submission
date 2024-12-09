"""
Transformer_RootInference.py

This script sets up and runs a transformer-based model for root inference tasks, trained on the full hierarchical data and tested on all filtering levels. This script assumes the data has been generated using gen_filtered_hierarchical_data.py.

It includes data preparation, model training, and evaluation.

Modules:
- numpy: For numerical operations.
- torch: For building and training the neural network.
- sys, os: For system operations and path management.
- training: Custom module for training the model.
- models: Custom module containing the TransformerClassifier model.
- logger_tree_language: Custom module for logging.

Key Variables:
- device: Specifies whether to use GPU or CPU for computations.
- seeds: List of random seeds for reproducibility.
- P: Array of training sample sizes.
- N_test: Number of test samples.
- sigma, q, l: Data parameters.
- loss_fn: Loss function for training.
- num_epochs: Number of training epochs.
- embedding_size: Size of the embedding layer in the model.
- n_layers: List specifying the number of layers in the transformer model.
"""

import numpy as np 
import torch
from torch import nn
import sys
import os

sys.path.append('../modules/')

from training import train_model
from models import TransformerClassifier
from logger_tree_language import get_logger

print('Imports done.',flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Running on device:', device,flush=True)

# Data params
seeds = [0]
sigma = 1.0 
q = 4
l = 4
# Training params
P = 2**np.arange(9,18)[-1:] # Number of training samples that will be used
N_test = int(1e4) # Number of test samples
loss_fn = nn.CrossEntropyLoss()
num_epochs = 200
embedding_size = 128
n_layers = [4]
n_head = 1

checkpoint_epochs = np.logspace(0,np.log10(num_epochs),12).astype(int)
checkpoint_epochs = np.unique(checkpoint_epochs).tolist()

# Load the full data
k = 0
[q,l,sigma,x0s,xis,M_s] = np.load('./data/labeled_data_factorized_{}_{}_{}_{}.npy'.format(q,l,sigma,k),allow_pickle=True)

# Will be used to compute validation on factorized models
factorized_layers = np.flip(np.arange(1,l+1))

for n_layer in n_layers:
    for seed in seeds:
        # Load the full data
        x0 = x0s[:,seed]
        xi = xis[:,:,seed]
        # Prepare the test data
        y_test = nn.functional.one_hot(torch.from_numpy(x0[-N_test:]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
        x_test = torch.from_numpy(xi[:,-N_test:].T).to(device=device).int()
        test_data_factorized = []
        # Prepare the test data on factorized trees
        for i in range(len(factorized_layers)):
            [_,_,_,x0s_factorized,xis_factorized,_,_] = np.load('./data/labeled_data_factorized_{}_{}_{}_{}.npy'.format(q,l,sigma,factorized_layers[i]),allow_pickle=True)
            x0_factorized = x0s_factorized[:,seed]
            xi_factorized = xis_factorized[:,:,seed]
            y_test_factorized = nn.functional.one_hot(torch.from_numpy(x0_factorized[-N_test:]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
            x_test_factorized = torch.from_numpy(xi_factorized[:,-N_test:].T).to(device=device).int()
            test_data_factorized.append((x_test_factorized,y_test_factorized))
        # Train the model
        for p in P:
            # Create model directory if needed
            model_dir = './models/model_wfactorizedval_LinearReadout_Rebuttal_{}_{}_{:.2f}_{}_{}_{}'.format(q,l,sigma,seed,p,n_layer)
            if model_dir is not None:
                # Create model directory if it doesn't exist.
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
            # Run training
            torch.cuda.empty_cache()
            x_train = torch.from_numpy(xi[:,:p].T).to(device=device).int()
            y_train = nn.functional.one_hot(torch.from_numpy(x0[:p]).to(dtype=torch.int64), num_classes=q).to(dtype=torch.float32).to(device=device)
            model = TransformerClassifier(
                seq_len=int(2**l),
                embedding_size=embedding_size,
                n_tranformer_layers=n_layer,
                n_heads=n_head,
                vocab_size=q,
                embedding_agg='flatten',
                positional_encoding=True,
                decoder_hidden_sizes=[],
            ).to(device=device)
            _, training_history = train_model(
                model=model,
                training_data=(x_train, y_train),
                test_data=(x_test, y_test),
                n_epochs=num_epochs,
                loss_fn=loss_fn,
                learning_rate=1e-4,
                batch_size=32,
                early_stopper=None,
                test_data_factorized=test_data_factorized,
                model_dir=model_dir,
                checkpoint_epochs=checkpoint_epochs
            )
            # Save the training history and settings
            np.save('./results/Transformer_RootInference_wfactorizedval_LinearReadout_Rebuttal_{}_{}_{:.2f}_{}_{}_{}.npy'.format(q,l,sigma,seed,p,n_layer),np.array([q,l,sigma,seed,p,n_layer,training_history,embedding_size],dtype=object))
            # Save the actual model for further finetuning and studying the attention maps
            checkpoint_id = 'model'
            checkpoint_path = os.path.join(model_dir,checkpoint_id + f'_epoch_{num_epochs}.pt')
            torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'training_history': training_history
                    },
                    checkpoint_path
                )