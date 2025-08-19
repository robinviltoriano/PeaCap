import os
import json
import matplotlib.pyplot as plt

import numpy as np

def plot_training_metrics(root_path:str, model_name:str, file:str) -> None:
    """
    This function reads the training metrics from a JSON file and plots in a 2x2 grid
    
    The training metrics should include:
    - Loss
    - Learning Rate
    - Gradient Norm
    
    Args:
        root_path (str): The root folder of the training results. It will also be the root folder for the plots.
        file (str): The name of the JSON file containing the training metrics.
        
    Returns:
        None: The function saves the plots as a JPG file in the specified root path.
    """
    training_results = json.load(open(os.path.join(root_path, file), 'r'))
    
    # Extracting metrics from the training results
    losses = training_results['loss']
    learning_rates = training_results['lr']
    grad_norms = training_results['grad_norm']
    log_grad_norms = np.log(grad_norms)
    
    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    axs[0, 0].plot(losses, label='Loss', color='blue')
    axs[0, 0].set_title('Training Loss Over Time')
    axs[0, 0].set_xlabel('Training Steps')
    axs[0, 0].legend()

    axs[0, 1].plot(grad_norms, label='Gradient Norm', color='orange')
    axs[0, 1].set_title('Training Gradient Norm Over Time')
    axs[0, 1].set_xlabel('Training Steps')
    axs[0, 1].legend()

    axs[1, 0].plot(losses, label='Loss', color='blue')
    axs[1, 0].plot(log_grad_norms, label='Log Gradient Norm', color='orange')
    axs[1, 0].set_title('Training Loss and Log of Gradient Norm Over Time')
    axs[1, 0].set_xlabel('Training Steps')
    axs[1, 0].legend()

    axs[1, 1].plot(learning_rates, label='Learning Rate', color='green')
    axs[1, 1].set_title('Learning Rate Over Time')
    axs[1, 1].set_xlabel('Training Steps')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].legend()
    
    model_name = model_name.split('.')[0] if '.' in model_name else model_name
    fig.suptitle(f'Training Metrics for {model_name}', fontsize=16)

    fig.savefig(os.path.join(root_path,'training_metrics_plots.jpg'), format='jpg')
