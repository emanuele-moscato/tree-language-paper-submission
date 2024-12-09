import os
import pickle
import pandas as pd
import torch
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error)
import matplotlib.pyplot as plt
import seaborn as sns
from logger_tree_language import get_logger


sns.set_theme()


def compute_accuracy(predicted_probs, one_hot_encoded_targets):
    """
    Computes the accuracy of the predictions,
        [n correct predictions] / [n samples]
    from the probabilities predicted by the model and the one-hot encoded
    target labels. Predicted labels are extracted via argmax.
    """
    return (
        torch.argmax(predicted_probs, axis=-1)
        == torch.argmax(one_hot_encoded_targets, axis=-1)
    ).to(dtype=torch.float32).mean()


def evaluate_torch_model(model, data, targets, target_scaler=None):
    """
    Given a model and (validation) data and targets, computes MSE, MAE and 
    MAPE and returns a dataframe with predictions, targets and residuals.
    """
    # Compute predictions and cast prediction and targets as rank-1 NumPy
    # arrays.
    with torch.no_grad():
        pred = model(data).detach().numpy()
        targets = targets.numpy()

    if target_scaler is not None:
        pred = target_scaler.inverse_transform(pred).ravel()
        targets = target_scaler.inverse_transform(targets).ravel()
    else:
        pred = pred.ravel()
        targets = targets.ravel()

    # Compute metrics.
    mse = mean_squared_error(targets, pred)
    mae = mean_absolute_error(targets, pred)
    mape = mean_absolute_percentage_error(targets, pred)

    print(f'MSE: {mse} | MAE: {mae} | MAPE: {mape}')

    results = pd.DataFrame({
        'pred': pred,
        'target': targets
    })

    results['residual'] = results['pred'] - results['target']
    results['relative_residual'] = results['residual'] / results['target']

    return mse, mae, mape, results


# def evaluate_sklearn_model(
#         model,
#         data,
#         targets,
#         bin_edges=None,
#         target_scaler=None
#     ):
#     """
#     """
#     if (bin_edges is None) and (target_scaler is None):
#         raise Exception('Either use target quantization or target rescaling')

#     if bin_edges is not None:
#         # Predictions are quantized: map them back to continuous values.
#         pred = map_classes_to_values(model.predict(data), bin_edges)
#     else:
#         pred = model.predict(data)

#         if target_scaler is not None:
#             pred = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
#             targets = target_scaler.inverse_transform(
#                 targets.reshape(-1, 1)
#             ).flatten()

#     flat_targets = targets.ravel()

#     mse = mean_squared_error(flat_targets, pred)
#     mae = mean_absolute_error(flat_targets, pred)
#     mape = mean_absolute_percentage_error(flat_targets, pred)

#     print(f'MSE: {mse} | MAE: {mae} | MAPE: {mape}')

#     results = pd.DataFrame({
#         'pred': pred,
#         'target': flat_targets
#     })

#     results['residual'] = results['pred'] - results['target']
#     results['relative_residual'] = results['residual'] / results['target']

#     return mse, mae, mape, results


# def cross_val_torch(
#         data,
#         targets,
#         continuous_features,
#         categorical_features,
#         selected_target,
#         data_kwargs,
#         model_kwargs,
#         optimizer_kwargs,
#         training_kwargs,
#         test_size=0.2,
#         early_stopper_kwargs=None
#     ):
#     """
#     Performs k-fold cross validation with a PyTorch model.
#     """
#     logger = get_logger(name='cross_val_torch')

#     n_iter = int(1. / test_size)

#     eval_metrics = []

#     logger.info(f'Performing {n_iter}-fold cross validation')

#     for i in range(n_iter):
#         # Split training and validation data.
#         training_data, val_data, training_target, val_target = (
#             get_train_val_data(
#                 data,
#                 targets[selected_target],
#                 test_size=test_size,
#                 stratification=data_kwargs['stratification']
#             )
#         )

#         # Preprocess training and validation data.
#         # Preprocess the training data (encoders/scalers are fit on this).
#         logger.debug('Preprocessing training data')

#         (
#             preprocessed_training_data,
#             rescaled_training_target,
#             encoder,
#             scaler,
#             target_scaler
#         ) = preprocess_data(
#             data=training_data,
#             target_values=training_target,
#             selected_target=selected_target,
#             categorical_features=categorical_features,
#             continuous_features=continuous_features,
#             target_range=(
#                 targets[selected_target].min(), targets[selected_target].max()
#             )
#         )

#         # Preprocess the validation data (using the encoders/scalers
#         # fit on the training data).
#         logger.debug('Preprocessing validation data')

#         preprocessed_val_data, rescaled_val_target, _, _, _ = preprocess_data(
#             data=val_data,
#             target_values=val_target,
#             selected_target=selected_target,
#             categorical_features=categorical_features,
#             continuous_features=continuous_features,
#             pretrained_encoder=encoder,
#             pretrained_scaler=scaler,
#             pretrained_target_scaler=target_scaler
#         )

#         # Instantiate model.
#         logger.debug('Instantiating model')

#         # Add the number of features to the `dims` parameter passed to the
#         # model. This is needed given the definition of the model class and
#         # can only be done after preprocessing because we don't know how many
#         # features will be there before one-hot encoding, plus it must be done
#         # ONLY ONCE!
#         if i == 0:
#             model_kwargs['dims'] = (
#                 [preprocessed_training_data.shape[1]] + model_kwargs['dims']
#             )

#         nn_model = FFNN(**model_kwargs)

#         if early_stopper_kwargs is not None:
#             early_stopper = EarlyStopper(**early_stopper_kwargs)
#         else:
#             early_stopper = None

#         # Train model.
#         training_history = train_nn_model(
#             training_data=preprocessed_training_data,
#             training_target=rescaled_training_target,
#             val_data=preprocessed_val_data,
#             val_target=rescaled_val_target,
#             nn_model=nn_model,
#             loss_fn=torch.nn.MSELoss(),
#             optimizer=torch.optim.Adam(
#                 params=nn_model.parameters(),
#                 **optimizer_kwargs
#             ),
#             early_stopper=early_stopper,
#             **training_kwargs
#         )

#         eval_mse, eval_mae, eval_mape, eval_results = evaluate_torch_model(
#             nn_model,
#             preprocessed_val_data,
#             rescaled_val_target,
#             target_scaler=target_scaler
#         )

#         eval_metrics.append({
#             'cv_iteration': i,
#             'eval_mse': eval_mse,
#             'eval_mae': eval_mae,
#             'eval_mape': eval_mape
#         })

#     eval_metrics = pd.DataFrame(eval_metrics)

#     aggregated_eval = pd.merge(
#         left=(
#             eval_metrics[['eval_mse', 'eval_mae', 'eval_mape']]
#             .mean()
#             .reset_index()
#             .rename(columns={'index': 'metric', 0: 'mean'})
#         ),
#         right=(
#             eval_metrics[['eval_mse', 'eval_mae', 'eval_mape']]
#             .std()
#             .reset_index()
#             .rename(columns={'index': 'metric', 0: 'st_dev'})
#         ),
#         on='metric',
#         how='inner'
#     )

#     return eval_metrics, aggregated_eval


# def list_experiment_ids(dir_path):
#     """
#     Lists the IDs of all the experiment recorded in the
#     directory `dir_path`.
#     """
#     return [
#         int(filename.split('expid_')[-1].split('.')[0])
#         for filename in os.listdir(dir_path)
#         if ('expid_' in filename) and ('.pkl' in filename)
#     ]


# def generate_experiment_id(dir_path):
#     """
#     """
#     expids = list_experiment_ids(dir_path)

#     if not expids:
#         return 1
#     else:
#         return max(expids) + 1


# def generate_experiment_filename(expid):
#     """
#     Filename convention: expid_{expid}.pkl
#     """
# #     return f'expid_{expid}.pkl'


def save_experiment_info_old(
        output_dir,
        data_version,
        continuous_features,
        categorical_features,
        selected_target,
        data_kwargs,
        model_kwargs,
        optimizer_kwargs,
        early_stopper_kwargs,
        training_kwargs,
        aggregated_eval,
    ):
    """
    """
    logger = get_logger(name='save_experiment_info')

    experiment_id = generate_experiment_id(output_dir)
    
    # Collect all data into a single (nested) dictionary.
    experiment_info = {
        'experiment_id': experiment_id,
        'dataset_info': {
            'data_version': data_version,
            'continuous_features': continuous_features,
            'categorical_features': categorical_features,
            'selected_target': selected_target
        },
        'data_params': data_kwargs,
        'model_params': model_kwargs,
        'optimizer_params': optimizer_kwargs,
        'early_stopper_params': early_stopper_kwargs,
        'training_params': training_kwargs,
        'aggregated_eval': aggregated_eval.to_dict(orient='records')
    }

    # Save the experiment info locally.
    output_path = os.path.join(
        output_dir,
        generate_experiment_filename(experiment_id)
    )

    with open(output_path, 'wb') as f:
        pickle.dump(experiment_info, f)

    # Add experiment data to the catalog.
    catalog_file_path = os.path.join(output_dir, 'experiment_catalog.csv')

    catalog_entry = pd.DataFrame([{
        f'{aggregated_eval.iloc[row]["metric"]}_{col}': (
            aggregated_eval.iloc[row][col]
        )
        for row in range(aggregated_eval.shape[0])
        for col in aggregated_eval.drop(columns=['metric']).columns
    }])
    
    catalog_entry['experiment_id'] = experiment_id
    catalog_entry['experiment_info_filepath'] = os.path.abspath(output_path)

    if not os.path.exists(catalog_file_path):
        catalog = catalog_entry
    else:
        catalog = pd.read_csv(catalog_file_path)

        catalog = pd.concat([catalog, catalog_entry])

    catalog.to_csv(catalog_file_path, header=True, index=None)

    logger.info(
        f'Experiment info saved in: {output_path} | '
        f'Added to catalog: {catalog_file_path}'
    )


def load_experiment_catalog(experiment_catalog_path):
    """
    """
    experiment_catalog = pd.read_csv(experiment_catalog_path)

    experiment_catalog['dims'] = (
        experiment_catalog['dims']
        .apply(
            lambda x: x.lstrip('[').rstrip(']').replace(' ', '').split(',')
        )
        .apply(lambda l: [int(el) for el in l])
    )

    return experiment_catalog


def save_experiment_info(experiment_catalog_path, **kwargs):
    """
    """
    logger = get_logger('experiment_manager')

    if not os.path.exists(experiment_catalog_path):
        logger.info(f'Creating experiment catalog: {experiment_catalog_path}')

        experiment_catalog = None
    else:
        experiment_catalog = pd.read_csv(experiment_catalog_path)

    new_experiment = pd.DataFrame([kwargs])

    if 'experiment_id' not in new_experiment.columns:
        raise Exception('No experiment_id column found for new experiment')

    if experiment_catalog is None:
        experiment_catalog = new_experiment
    else:
        exp_id = new_experiment['experiment_id'].iloc[0]

        if exp_id in experiment_catalog['experiment_id'].tolist():
            raise Exception(
                f'The specified experiment ID {exp_id} already exists'
            )

        experiment_catalog = pd.concat(
            [experiment_catalog, new_experiment]
        ).reset_index(drop=True)

    experiment_catalog.to_csv(experiment_catalog_path, index=None)
