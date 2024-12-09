import os
import logging
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter
from logger_tree_language import get_logger
from models import switch_training_mode_submodules


def mask_sequences(
        sequences,
        mask_rate,
        reshaped_mask_idx,
        device,
        single_mask=False,
        src_key_padding_mask=None
    ):
    """
    Performs random masking of the elements of the input sequence,
    substituting them with the token in `reshaped_mask_idx` with probability
    `mask_rate`. `reshaped_mask_idx` must be a tensor with the same shape as
    `sequences` containing the mask token only. If `single_mask` is True,
    only and exactly one symbol per sequence is masked.

    If working with padded sequences, pass the padding mask (same as the one
    used for the model's forward pass) to `src_key_padding_mask` so the
    padding tokens are ignored: all the OTHER tokens in the sequences will
    have probability `mask_rate` of being masked (or only one of the other
    tokens will be masked, if `single_mask` is True).
    """
    # Generate a boolean mask of shape `sequences.shape` with elements
    # having a `mask_rate` probability of being True (corresponding to
    # the elements to mask).
    mask = (torch.rand(size=sequences.shape) < mask_rate).to(device=device)

    if single_mask: # Only one mask per sequence
        mask = torch.zeros(sequences.shape, dtype=torch.bool, device=device)

        if src_key_padding_mask is not None:
            # Generate a random index for each sequence in the batch,
            # excluding the padding tokens (always mask one of the other
            # tokens).
            # Note: couldn't find a way to do that without a for loop, do we
            #       need this?
            raise NotImplementedError(
                "Random masking of a single symbol per sequence hasn't been"
                " implemented yet for padded sequences"
            )
        else:
            # Generate a random index for each sequence in the batch
            random_indices = torch.randint(0, sequences.shape[-1], (sequences.shape[0],), device=device)
            mask[torch.arange(sequences.shape[0]), random_indices] = True

    # Mask the sequences: replace the elements corresponding to the True
    # entries of the mask with the `mask_idx` index.
    # Note: if padding tokens are there, this "masks" them as well (they are
    #       reintroduced below).
    masked_sequences = torch.where(
        mask,
        reshaped_mask_idx,
        sequences
    )

    # If using padded sequences, reintroduce the padding token in all the
    # padded positions, so at the end none of them is turned into the mask
    # token.
    if src_key_padding_mask is not None:
        masked_sequences = torch.where(
            src_key_padding_mask,
            sequences,
            masked_sequences
        )
    
    return masked_sequences, mask


def training_step_mlm(batch, masked_batch, mask, model, loss_fn, optimizer):
    """
    Performs a training step for masked language modeling. The predictions are
    computed for the batch (both masked and non-masked tokens), as well as the
    loss (a tensor of values), then we average over the loss values
    corresponding to the masked tokens only.
    """
    # Compute predictions on the masked batch of sequences.
    pred = model(masked_batch)
    
    # Compute the average loss between the predicted masked tokens and the
    # actual ones in the masked positions.
    training_loss = loss_fn(
        torch.permute(
            pred,
            (0, 2, 1)),
        batch
    )[mask].mean()

    # Reset gradients and recompute it.
    optimizer.zero_grad()

    training_loss.backward()

    # Perform an optimization step.
    optimizer.step()
    
    # return training_loss.detach().numpy(), val_loss
    return training_loss.detach()


def compute_masked_accuracy(pred_logits, actual_sequence, mask):
    """
    Compute the reconstruction accuracy on the masked tokens.
    """
    return (
        torch.argmax(pred_logits, axis=-1) == actual_sequence
    )[mask].to(dtype=torch.float32).mean()


def train_model_mlm(
        sequences,
        model,
        n_epochs,
        batch_size,
        mask_rate,
        mask_idx,
        device,
        optimizer,
        lr_schedule_fn=None,
        training_history=None,
        checkpointing_period_epochs=None,
        model_dir=None,
        checkpoint_id=None,
        tensorboard_log_dir=None,
        val_sequences=None,
        single_mask=False,
        test_data_factorized=None,
        frozen_encoder=False,
        checkpoint_times=None
    ):
    """
    Trains a model for masked language modeling.

    If `test_data_factorized` is passed, it works as a list of additional
    validation datasets (not necessarily with the same number of sequences),
    with `test_data_factorized[i]` being the tensor of sequences for dataset
    `i`.

    Validation metrics are computed by putting the model in evaluation mode
    (i.e. with dropout and batch normalization - if any - layers deactivated).
    If `frozen_encoder` is `True`, only the `decoder` submodule is switched
    between training and evaluation mode when computing validation metrics.
    """
    logger = get_logger('train_model_mlm', level=logging.INFO)

    logger.info('Training model')

    update_counter = 0

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

        if test_data_factorized is not None:
            training_history['val_loss_factorized'] = []
            training_history['val_accuracy_factorized'] = []

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

    training_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(sequences),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # Broadcast `mask_idx` to the same shape of a batch of sequences and send
    # it to the selected device.
    reshaped_mask_idx = (
        mask_idx.repeat(sequences[:batch_size, ...].shape).to(device=device)
    )

    if val_sequences is not None:
        reshaped_mask_idx_val = (
            mask_idx.repeat(val_sequences.shape).to(device=device)
        )

    if test_data_factorized is not None:
        # Prepare masks with the appropriate shape for each of the
        # factorized dataset (this way they can contain different numbers)
        # of sequences.
        reshaped_mask_idx_factorized_datasets = [
            mask_idx.repeat(factorized_dataset.shape).to(device=device)
            for factorized_dataset in test_data_factorized
        ]

    # Evaluate the model on validation and/or factorized data before the frist
    # epoch to get the untrained performance (only if starting training from
    # scratch).
    if epoch_counter == 0:
        # Evaluate performance of the validation data.
        if val_sequences is not None:
            with torch.no_grad():
                # Randomly mask the validation data.
                masked_validation_sequences, mask_val = mask_sequences(
                    val_sequences,
                    mask_rate=mask_rate,
                    reshaped_mask_idx=reshaped_mask_idx_val,
                    device=device,
                    single_mask=single_mask
                )

                # Generate predictions for the validation data, switching to
                # evaluation mode.
                model.eval()

                val_pred = model(masked_validation_sequences)

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
                val_loss = loss_fn(
                    torch.permute(
                        val_pred,
                        (0, 2, 1)),
                    val_sequences
                )[mask_val].mean()

                val_accuracy = compute_masked_accuracy(
                    val_pred,
                    val_sequences,
                    mask_val
                )

            training_history['val_loss'].append(val_loss)
            training_history['val_accuracy'].append(val_accuracy)

        # Evaluate performance on the factorized sequences.
        if test_data_factorized is not None:
            factorized_losses = []
            factorized_accuracies = []

            # Loop over the factorized datasets (sequences).
            for i, factorized_sequences in enumerate(test_data_factorized):
                with torch.no_grad():
                    # Randomly mask the factorized data.
                    # `factorized_dataset` is assumed to be a tensor of
                    # sequences generated by a factorized generative
                    # model.
                    masked_factorized_sequences, mask_factorized = mask_sequences(
                        factorized_sequences,
                        mask_rate=mask_rate,
                        reshaped_mask_idx=reshaped_mask_idx_factorized_datasets[i],
                        device=device,
                        single_mask=single_mask
                    )

                    # Generate predictions for the validation data, switching to
                    # evaluation mode.
                    model.eval()

                    factorized_pred = model(masked_factorized_sequences)

                    # If the encoder must be kept frozen (and, by default, in
                    # evaluation mode), put only the decoder back into training
                    # mode, otherwise the whole model.
                    if frozen_encoder:
                        switch_training_mode_submodules(
                            'train', model, ['decoder']
                        )
                    else:
                        model.train()

                    # Compute (masked) loss and accuracy on the factorized
                    # data.
                    factorized_loss = loss_fn(
                        torch.permute(
                            factorized_pred,
                            (0, 2, 1)),
                        factorized_sequences
                    )[mask_factorized].mean()

                    factorized_accuracy = compute_masked_accuracy(
                        factorized_pred,
                        factorized_sequences,
                        mask_factorized
                    )

                factorized_losses.append(factorized_loss)
                factorized_accuracies.append(factorized_accuracy)

            training_history['val_loss_factorized'].append(factorized_losses)
            training_history['val_accuracy_factorized'].append(factorized_accuracies)

    # Training loop.
    with trange(n_epochs) as pbar:
        for _ in pbar:
            epoch_counter += 1

            training_loss_batches = []
            training_accuracy_batches = []

            for batch in training_loader:
                update_counter += 1

                # In this case, batch is a 1-element list (the batch of
                # sequences).
                batch = batch[0]

                # Perform random masking on the batch.
                masked_batch, mask = mask_sequences(
                    batch,
                    mask_rate,
                    reshaped_mask_idx,
                    device,
                    single_mask
                )

                training_loss_batch = training_step_mlm(
                    batch,
                    masked_batch,
                    mask,
                    model,
                    loss_fn,
                    optimizer
                )

                training_loss_batches.append(training_loss_batch)

                # Compute the training accuracy over the batch and append it
                # to the corresponding list.
                training_accuracy_batch = compute_masked_accuracy(
                    model(masked_batch),
                    batch,
                    mask
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
                    # Randomly mask the validation data.
                    masked_validation_sequences, mask_val = mask_sequences(
                        val_sequences,
                        mask_rate=mask_rate,
                        reshaped_mask_idx=reshaped_mask_idx_val,
                        device=device,
                        single_mask=single_mask
                    )

                    # Generate predictions for the validation data, switching
                    # to evaluation mode.
                    model.eval()

                    val_pred = model(masked_validation_sequences)

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
                    val_loss = loss_fn(
                        torch.permute(
                            val_pred,
                            (0, 2, 1)),
                        val_sequences
                    )[mask_val].mean()

                    val_accuracy = compute_masked_accuracy(
                        val_pred,
                        val_sequences,
                        mask_val
                    )

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

            # If test factorized data is passed, compute the validation
            # metrics on the factorized datasets.
            if test_data_factorized is not None:
                factorized_losses = []
                factorized_accuracies = []

                # Loop over the factorized datasets (sequences).
                for i, factorized_sequences in enumerate(test_data_factorized):
                    with torch.no_grad():
                        # Randomly mask the factorized data.
                        # `factorized_dataset` is assumed to be a tensor of
                        # sequences generated by a factorized generative
                        # model.
                        masked_factorized_sequences, mask_factorized = mask_sequences(
                            factorized_sequences,
                            mask_rate=mask_rate,
                            reshaped_mask_idx=reshaped_mask_idx_factorized_datasets[i],
                            device=device,
                            single_mask=single_mask
                        )

                        # Generate predictions for the validation data, switching to
                        # evaluation mode.
                        model.eval()

                        factorized_pred = model(masked_factorized_sequences)

                        # If the encoder must be kept frozen (and, by default,
                        # in evaluation mode), put only the decoder back into
                        # training mode, otherwise the whole model.
                        if frozen_encoder:
                            switch_training_mode_submodules(
                                'train', model, ['decoder']
                            )
                        else:
                            model.train()

                        # Compute (masked) loss and accuracy on the factorized
                        # data.
                        factorized_loss = loss_fn(
                            torch.permute(
                                factorized_pred,
                                (0, 2, 1)),
                            factorized_sequences
                        )[mask_factorized].mean()

                        factorized_accuracy = compute_masked_accuracy(
                            factorized_pred,
                            factorized_sequences,
                            mask_factorized
                        )

                    factorized_losses.append(factorized_loss)
                    factorized_accuracies.append(factorized_accuracy)

                training_history['val_loss_factorized'].append(factorized_losses)
                training_history['val_accuracy_factorized'].append(factorized_accuracies)

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
    
    if test_data_factorized is not None:
        training_history['val_loss_factorized'] = torch.tensor(training_history['val_loss_factorized']).tolist()
        training_history['val_accuracy_factorized'] = torch.tensor(training_history['val_accuracy_factorized']).tolist()


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
