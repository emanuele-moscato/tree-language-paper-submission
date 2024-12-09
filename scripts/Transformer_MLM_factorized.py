"""
Transformer_MLM_factorized.py

This script sets up and runs a masked language model (MLM) using a transformer architecture trained on all filtering levels and tested on the full hierarchy. This script assumes the data has been generated using gen_filtered_hierarchical_data.py.

It includes data preparation, model training, and evaluation.

Modules:
- os, sys: For system operations and path management.
- json: For handling JSON data.
- numpy: For numerical operations.
- torch: For building and training the neural network.
- logger_tree_language: Custom module for logging.
- models: Custom module containing the TransformerClassifier model.
- masked_language_modeling: Custom module for training the MLM and masking sequences.
- pytorch_utilities: Custom module for learning rate schedulers and updates.

Key Variables:
- logger: Logger for tracking the training process.
- device: Specifies whether to use GPU or CPU for computations.

Functions:
- setup(seq_len, embedding_size, vocab_size, n_layer, n_head, device, q, l, sigma, epsilon, seed, p, mask_frac): 
  Sets up the transformer model with the given parameters.

"""

import os
import sys
import json
import numpy as np
import torch

sys.path.append('../modules/')

from logger_tree_language import get_logger
from models import TransformerClassifier
from masked_language_modeling import train_model_mlm
from pytorch_utilities import create_linear_lr_schedulers, lr_linear_update

logger = get_logger('transformer_encoder_pretraining')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Running on device:', device,flush=True)

def setup(seq_len,embedding_size,vocab_size,n_layer,n_head,device,q,l,sigma,epsilon,seed,p,mask_frac,factorized_layer):
    # Instantiate model.
    model_params = dict(
        seq_len=seq_len,
        embedding_size=embedding_size,
        n_tranformer_layers=n_layer,  # Good: 4
        n_heads=n_head,
        vocab_size=vocab_size,
        #encoder_dim_feedforward=2 * embedding_size # Alternatively default is 2048
        positional_encoding=True,
        n_special_tokens=1,  # We assume the special tokens correspond to the last `n_special_tokens` indices.
        embedding_agg=None,
        decoder_hidden_sizes=[],  # Good: [64]
        decoder_activation='relu',  # Good: 'relu'
        decoder_output_activation='identity'
    )
    # Get the model
    model = TransformerClassifier(
        **model_params
    ).to(device=device)
    # Instantiate optimizer.
    optimizer_params = dict(
        lr=1e-4, # correcting for the very small loss due to single token masking
        betas=(0.9, 0.999),
        eps=1e-08,  # 1e-6 for BERT.
        weight_decay=0  # 0.01 for BERT.
    )
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        **optimizer_params
    )
    # Create a directory in which to save the train model (and checkpoints) and the
    # optimizer's final state.
    model_dir = './models/model_MLM_1mask_factorized_LinearReadout_{}_{}_{:.2f}_{}_{}_{}_{}'.format(q,l,sigma,seed,p,n_layer,factorized_layer)

    if model_dir is not None:
        # Create model directory if it doesn't exist.
        if not os.path.exists(model_dir):
            logger.info(f'Creating model directory: {model_dir}')

            os.makedirs(model_dir)

        # Save model's and optimizer's (hyper)parameters.
        model_params_path = os.path.join(model_dir, 'model_params.json')

        with open(model_params_path, 'w') as f:
            json.dump(model_params, f)
        
        optimizer_params_path = os.path.join(model_dir, 'optimizer_params.json')

        with open(optimizer_params_path, 'w') as f:
            json.dump(optimizer_params, f)

    # Set up learning rate scheduling, will actually not use it here but example of how to do it.
    batch_size = 32  # 32 is the standard (e.g. in Keras)
    n_epochs = int(2e3)
    warmup_updates_frac = 0.15

    n_updates_total = int(leaves.shape[0] / batch_size) * n_epochs
    n_updates_warmup = int(n_updates_total * warmup_updates_frac)
    n_updates_decay = n_updates_total - n_updates_warmup

    # Here will not use this lr_scheduler_fn, but it is an example of how to use it.
    lr_scheduler_warmup, lr_scheduler_decay = create_linear_lr_schedulers(optimizer, n_updates_warmup, n_updates_decay)
    lr_schedule_fn = lambda update_counter: lr_linear_update(update_counter, lr_scheduler_warmup, lr_scheduler_decay, n_updates_warmup)

    return model,optimizer,lr_schedule_fn,batch_size,n_epochs,model_dir

# Data params
seeds = [0]
sigma = 1.0
epsilon = 0.0
q = 4
l = 4
# Training params
n_layers = [4]
n_heads = [1]
embedding_size = 128
seq_len = 2**l
mask_frac = 0.5
vocab = torch.arange(q).to(dtype=torch.int64)
mask_idx = vocab.max() + 1
# Enlarge the vocabulary with the special tokens.
vocab = torch.hstack([vocab, torch.Tensor(mask_idx).to(dtype=torch.int64)])
vocab_size = vocab.shape[0]
N_val = int(1e4)
P = [2**18]
checkpoint_epochs = np.logspace(0,3,8).astype(int).tolist()

# Load the full data
k = 0
[q,l,sigma,_,xis,M_s] = np.load('./data/labeled_data_factorized_{}_{}_{}_{}.npy'.format(q,l,sigma,k),allow_pickle=True)

factorized_layers = np.flip(np.arange(1,l+1))

for n_layer in n_layers:
    for n_head in n_heads:
        for seed in seeds:
            for p in P:
                for factorized_layer in factorized_layers:
                    [_,_,_,_,xis_factorized,_,_] = np.load('./data/labeled_data_factorized_{}_{}_{}_{}.npy'.format(q,l,sigma,factorized_layer),allow_pickle=True)
                    torch.cuda.empty_cache() # Avoids running into memory issues
                    xi = xis_factorized[:,:,seed]
                    xi_val = xis[:,-N_val:,seed]
                    leaves = torch.from_numpy(xi[:,:p].T).to(device=device).to(dtype=torch.int64)
                    leaves_validation = torch.from_numpy(xi_val.T).to(device=device).to(dtype=torch.int64)
                    # Setup the learning
                    model,optimizer,lr_schedule_fn,batch_size,n_epochs,model_dir = setup(seq_len,embedding_size,vocab_size,n_layer,n_head,device,q,l,sigma,epsilon,seed,p,mask_frac,factorized_layer)
                    # Actually train
                    model, optimizer, training_history = train_model_mlm(
                        sequences=leaves,
                        model=model,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        mask_rate=mask_frac,
                        mask_idx=mask_idx,
                        device=device,
                        optimizer=optimizer,
                        #lr_schedule_fn=lr_schedule_fn,
                        checkpointing_period_epochs=500,
                        model_dir=model_dir,
                        checkpoint_id='model_',
                        #tensorboard_log_dir='../../tensorboard_logs/exp_4/'
                        val_sequences=leaves_validation,
                        single_mask=True
                    )
                    np.save('./results/Transformer_MLM_1mask_factorized_LinearReadout_{}_{}_{:.2f}_{}_{}_{}_{}.npy'.format(q,l,sigma,seed,p,n_layer,factorized_layer),np.array([q,l,sigma,seed,p,n_layer,training_history,embedding_size,p,N_val],dtype=object))