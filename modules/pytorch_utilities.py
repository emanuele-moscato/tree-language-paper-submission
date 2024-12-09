import os
import json
from traceback import format_exc
import torch
from models import TransformerClassifier


def count_model_params(model):
    """
    Counts the total number of parameters of a model, irrespective whether
    they are trainable or not.
    """
    return sum([p.numel() for p in model.parameters()])


def optimizer_to(optim, device):
    """
    Sends a PyTorch optimizer's parameters to the selected device,
    as it doesn't have a native `to` method. Useful when loading
    an optimizer's state from a state dict (`load_state_dict`
    method) to resume training on a GPU).
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def load_checkpoint(model_dir, checkpoint_id, device):
    """
    Loads a saved checkpoint (trained model, optimizer and training history)
    and sends the model's and optimizer's parameters to the specified device.
    """
    # Read saved model's (hyper) parameters.
    model_params_path = os.path.join(model_dir, 'model_params.json')
    
    with open(model_params_path, 'r') as f:
        model_params_loaded = json.load(f)
    
    optimizer_params_path = os.path.join(model_dir, 'optimizer_params.json')
    
    with open(optimizer_params_path, 'r') as f:
        optimizer_params_loaded = json.load(f)

    # Instantiate new model.
    model_loaded = TransformerClassifier(
        **model_params_loaded
    )
    
    # Instantiate new optimizer.
    optimizer_loaded = torch.optim.Adam(
        params=model_loaded.parameters(),
        **optimizer_params_loaded
    )
    
    # Load checkpoint.
    checkpoint_path = os.path.join(model_dir, checkpoint_id)
    
    checkpoint = torch.load(checkpoint_path)
    
    # Load data from the checkpoint into the model/optimizer/other variables.
    model_loaded.load_state_dict(checkpoint['model_state_dict'])

    # Send loaded model to the chosen device.
    model_loaded = model_loaded.to(device=device)

    try:
        optimizer_loaded.load_state_dict(checkpoint['optimizer_state_dict'])

        # Send loaded optimizer to the chosen device.
        optimizer_to(optimizer_loaded, device)
    except Exception as e:
        print("Couldn't load optimizer")
        print(format_exc())

        optimizer_loaded = None
    try:
        training_history_loaded = checkpoint['training_history']
    except Exception as e:
        print("Couldn't load the training history")
        print(format_exc())

        training_history_loaded = None

    return model_loaded, optimizer_loaded, training_history_loaded


def create_linear_lr_schedulers(
        optimizer,
        n_updates_warmup,
        n_updates_decay,
        warmup_start_factor=0.1,
        decay_end_factor=1e-2
    ):
    """
    Returns the appropriate learning rate schedulers to perform a linear
    warmup for the first `n_updates_warmup` updates and a linear decay
    for the later `n_updates_decay` updates. Everything is thought in
    terms of the PEAK VALUE of the learning rate, which is the one passed
    at instantiation time to the optimizer, so that the learning rate
        * starts at `peak_lr * warmup_start_factor`,
        * reaches `peak_lr` linearly over the first `n_updates_warmup`
          updates,
        * reaches `peak_lr * decay_end_factor` linearly over the next
          `n_updates_decay` updates.
    """
    lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.,
        total_iters=n_updates_warmup,
        last_epoch=-1
    )
    
    lr_scheduler_decay = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.,
        end_factor=decay_end_factor,
        total_iters=n_updates_decay,
        last_epoch=-1
    )

    return lr_scheduler_warmup, lr_scheduler_decay


def lr_linear_update(
        update_counter,
        lr_scheduler_warmup,
        lr_scheduler_decay,
        n_updates_warmup
    ):
    """
    Performs an update of the optimizer's learning rate using
    `lr_scheduler_warmup` for the first `n_updates_warmup` updates
    and `lr_scheduler_decay` for all the later ones.
    """
    if update_counter < n_updates_warmup:
        lr_scheduler_warmup.step()
    else:
        lr_scheduler_decay.step()
