#!/usr/bin/env python
import os
import yaml
import argparse
import torch
import torchvision.utils as vutils
from pathlib import Path

from experiment import VAEXperiment
from models import vae_models

def main():
    """
    Main function to sample images from a trained VAE model.
    """
    parser = argparse.ArgumentParser(description='Sample images from a trained VAE model.')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='Path to the config file',
                        required=True)
    parser.add_argument('--checkpoint', '-ckpt',
                        metavar='FILE',
                        help='Path to the checkpoint file',
                        required=True)
    parser.add_argument('--num_samples',
                        type=int,
                        default=64,
                        help='Number of samples to generate (default: 64)')
    parser.add_argument('--class_id',
                        type=int,
                        required=True,
                        help='Class ID for conditional generation (e.g., 0 for cat, 1 for dog)')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results_sampling',
                        help='Directory to save the generated images (default: ./results_sampling)')
    parser.add_argument('--gpu_id',
                        type=int,
                        default=None,
                        help='ID of the GPU to use. If not specified, CPU will be used.')

    args = parser.parse_args()

    # --- Load Config ---
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return

    # --- Setup Device ---
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Load Model from Checkpoint ---
    # Initialize model structure from config
    model = vae_models[config['model_params']['name']](**config['model_params'])
    
    # Initialize experiment wrapper, which is what's saved in the checkpoint
    experiment = VAEXperiment(model, config['exp_params'])

    # Load the state from the checkpoint file
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        experiment.load_state_dict(checkpoint['state_dict'])
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Get the actual model from the experiment wrapper and set to evaluation mode
    model = experiment.model
    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --- Create Output Directory ---
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # --- Prepare Labels for Conditional Generation ---
    labels = torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)

    # --- Generate Samples ---
    print(f"Generating {args.num_samples} samples for class ID {args.class_id}...")
    with torch.no_grad():
        samples = model.sample(num_samples=args.num_samples,
                               current_device=device,
                               labels=labels)

    # --- Save Samples ---
    # Create a filename that includes model name, class, and number of samples
    model_name = config['model_params']['name']
    version = Path(args.checkpoint).parent.parent.name # Extract version from path
    filename = f"samples_{model_name}_{version}_class_{args.class_id}_n_{args.num_samples}.png"
    output_filepath = output_path / filename
    
    vutils.save_image(samples.cpu().data,
                      fp=str(output_filepath),
                      normalize=True,
                      nrow=int(args.num_samples**0.5))

    print(f"Saved generated images to {output_filepath}")

if __name__ == '__main__':
    main()
