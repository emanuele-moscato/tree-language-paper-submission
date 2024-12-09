import os
import logging
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
from logger_tree_language import get_logger
from models import switch_training_mode_submodules

def training_step_ancestor(
        training_data,
        model,
        loss_fn,
        optimizer
    ):
    """
    Implements a training step for `model` using the algorithm implemented
    in `optimizer` and minimizing the loss `loss_fn`.
    """
    # Unpack the training data.
    x_train,y_train = training_data

    # Compute the loss function on the training data.
    y_pred = model(x_train)
    training_loss = loss_fn(y_pred.permute(0,2,1), y_train).mean() # Standard loss on all elements of the sequence, uncomment below to just use the last and first tokens (i.e. to check it does not overfit)
    #training_loss = loss_fn(y_pred.permute(0,2,1), y_train).transpose(0,1) # dim should be sequence length, batch_size
    #training_loss = (torch.stack([training_loss[0], training_loss[-1]], dim=0)).mean()

    # Reset gradients and recompute it.
    optimizer.zero_grad()

    training_loss.backward()

    # Perform an optimization step.
    optimizer.step()

    return training_loss.detach()

def compute_ancestors_accuracy(pred_logits, actual_sequence):
    """
    Compute the ancestor of each of the tokens, NB here assume actual sequence is an array of integers (NOT one-hot encoded).
    """
    return (
        torch.argmax(pred_logits, axis=-1) == actual_sequence
    ).to(dtype=torch.float32).mean()


def train_model_ancestor_probe(
        sequences,
        model,
        n_epochs,
        batch_size,
        optimizer,
        lr_schedule_fn=None,
        training_history=None,
        checkpointing_period_epochs=None,
        model_dir=None,
        checkpoint_id=None,
        tensorboard_log_dir=None,
        val_sequences=None,
        frozen_encoder=False,
        checkpoint_times=None
    ):
    """
    Trains a model for ancestor (at any level) predictions.

    The labels (actual ancestors) are assumed to be one-hot encoded, and to be upscaled in dimension to be of the same dimension as the sequences. The output of the model is therefore a sequence of logits of the same dimension as the input sequences, each representing the probability of the corresponding token being the ancestor of the token at the same position in the input sequence.
    """
    logger = get_logger('train_model_ancestor_probe', level=logging.INFO)

    logger.info('Training model')

    update_counter = 0

    print('Reloaded')

    if training_history is None:
        epoch_counter = 0

        training_history = {
            'training_loss': [],
            'training_accuracy': [],
            'learning_rate': [],
            'learning_rate_updates': []
        }

        if val_sequences is not None:
            training_history['val_loss'] = []
            training_history['val_accuracy'] = []

    else:
        # Resume training from the last epoch, as inferred by the length of
        # the provided training history.
        epoch_counter = len(
            training_history[list(training_history.keys())[0]]
        )

        logger.info(f'Resuming training from epoch {epoch_counter}')

        if 'val_loss' in training_history.keys():
            if val_sequences is None:
                raise Exception(
                    'Validation data was used in previous training, please '
                    'keep using it'
                )
        else:
            if val_sequences is not None:
                raise Exception(
                    'No validation data was used in previous training, please'
                    'keep not using it'
                )

    if tensorboard_log_dir is not None:
        writer = SummaryWriter(
            log_dir=tensorboard_log_dir
        )
    else:
        writer = None

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='none'
    )

    x_train,y_train = sequences

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train,y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Evaluate the model on validation and/or factorized data before the frist
    # epoch to get the untrained performance (only if starting training from
    # scratch).
    if epoch_counter == 0:
        # Evaluate performance of the validation data.
        if val_sequences is not None:
            with torch.no_grad():
                # Generate predictions for the validation data, switching to
                # evaluation mode.
                model.eval()

                x_val,y_val = val_sequences

                val_pred = model(x_val)

                # If the encoder must be kept frozen (and, by default, in
                # evaluation mode), put only the decoder back into training
                # mode, otherwise the whole model.
                if frozen_encoder:
                    switch_training_mode_submodules(
                        'train', model, ['decoder']
                    )
                else:
                    model.train()

                # Compute validation (masked) loss and accuracy.
                #val_loss = loss_fn(val_pred.permute(0,2,1),y_val).mean()
                val_loss = loss_fn(val_pred.permute(0,2,1), y_val).transpose(0,1) # dim should be sequence length, batch_size
                val_loss = (torch.stack([val_loss[0], val_loss[-1]], dim=0)).mean()

                val_accuracy = compute_ancestors_accuracy(val_pred,y_val)

            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)

    # Training loop.
    with trange(n_epochs) as pbar:
        for _ in pbar:
            epoch_counter += 1

            training_loss_batches = []
            training_accuracy_batches = []

            for batch in training_loader:
                update_counter += 1

                x_batch,y_batch = batch

                # Perform a training step.
                training_loss_batch = training_step_ancestor(
                    (x_batch,y_batch),
                    model,
                    loss_fn,
                    optimizer
                )

                training_loss_batches.append(training_loss_batch)

                training_accuracy_batch = compute_ancestors_accuracy(
                    model(x_batch),
                    y_batch,
                )
                training_accuracy_batches.append(training_accuracy_batch)

                # Update the learning rate, if needed.
                if lr_schedule_fn is not None:
                    lr_schedule_fn(update_counter)

                    training_history['learning_rate_updates'].append(
                        optimizer.state_dict()['param_groups'][0]['lr']
                    )

            # Training loss and accuracy for one epoch is computed as the average
            # training loss over the batches.
            training_loss = torch.tensor(training_loss_batches).mean()
            training_accuracy = torch.tensor(training_accuracy_batches).mean()
            
            training_history['training_loss'].append(training_loss)
            training_history['training_accuracy'].append(training_accuracy)
            training_history['learning_rate'].append(
                optimizer.state_dict()['param_groups'][0]['lr']
            )

            # If valdiation data is passed as input, compute the (masked) loss
            # and accuracy.
            if val_sequences is not None:
                with torch.no_grad():
                    # Generate predictions for the validation data, switching
                    # to evaluation mode.
                    model.eval()

                    x_val,y_val = val_sequences

                    val_pred = model(x_val)

                    # If the encoder must be kept frozen (and, by default, in
                    # evaluation mode), put only the decoder back into
                    # training mode, otherwise the whole model.
                    if frozen_encoder:
                        switch_training_mode_submodules(
                            'train', model, ['decoder']
                        )
                    else:
                        model.train()

                    # Compute validation (masked) loss and accuracy.
                    #val_loss = loss_fn(val_pred.permute(0,2,1),y_val).mean()
                    val_loss = loss_fn(val_pred.permute(0,2,1), y_val).transpose(0,1) # dim should be sequence length, batch_size
                    val_loss = (torch.stack([val_loss[0], val_loss[-1]], dim=0)).mean()

                    val_accuracy = compute_ancestors_accuracy(val_pred,y_val)

                training_history['val_loss'].append(val_loss)
                training_history['val_accuracy'].append(val_accuracy)

                pbar.set_postfix(
                    training_loss=training_history['training_loss'][-1],
                    training_accuracy=training_history['training_accuracy'][-1],
                    val_loss=training_history['val_loss'][-1],
                    val_accuracy=training_history['val_accuracy'][-1],
                    learning_rate=training_history['learning_rate'][-1]
                )
            else:
                pbar.set_postfix(
                    training_loss=training_history['training_loss'][-1],
                    training_accuracy=training_history['training_accuracy'][-1],
                    learning_rate=training_history['learning_rate'][-1]
                )

            # Write scalars to Tensorboard logs.
            if writer is not None:
                writer.add_scalar(
                    'Loss/train',
                    training_history['training_loss'][-1],
                    epoch_counter
                )
                writer.add_scalar(
                    'Accuracy/train',
                    training_history['training_accuracy'][-1],
                    epoch_counter
                )
                writer.add_scalar(
                    'LR/train',
                    training_history['learning_rate'][-1],
                    epoch_counter
                )

                if val_sequences is not None:
                    writer.add_scalar(
                        'Loss/val',
                        training_history['val_loss'][-1],
                        epoch_counter
                    )
                    writer.add_scalar(
                        'Accuracy/val',
                        training_history['val_accuracy'][-1],
                        epoch_counter
                    )

            if (
                (checkpointing_period_epochs is not None)
                and (epoch_counter % checkpointing_period_epochs == 0)
            ) or (checkpoint_times is not None and epoch_counter in checkpoint_times):
                # Save model/optimizer checkpoint.
                checkpoint_path = os.path.join(
                    model_dir, 
                    checkpoint_id + f'_epoch_{epoch_counter}.pt'
                )

                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_history': training_history
                    },
                    checkpoint_path
                )

    training_history['training_loss'] = torch.tensor(training_history['training_loss']).tolist()
    training_history['training_accuracy'] = torch.tensor(training_history['training_accuracy']).tolist()

    if val_sequences is not None:
        training_history['val_loss'] = torch.tensor(training_history['val_loss']).tolist()
        training_history['val_accuracy'] = torch.tensor(training_history['val_accuracy']).tolist()

    logger.info(f'Last epoch: {epoch_counter}')

    if checkpointing_period_epochs is not None:
        logger.info('Saving final model checkpoint')

        # Save model/optimizer checkpoint.
        checkpoint_path = os.path.join(
            model_dir, 
            checkpoint_id + f'_epoch_{epoch_counter}.pt'
        )

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': training_history
            },
            checkpoint_path
        )

    if writer is not None:
        writer.flush()
        writer.close()

    return model, optimizer, training_history