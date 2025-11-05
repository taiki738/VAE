import argparse
import yaml
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from models import *
from experiment import VAEXperiment
from dataset import VAEDataset
import pandas as pd

def analyze(args):
    # Load config
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return

    # For reproducibility
    torch.manual_seed(config['exp_params']['manual_seed'])
    np.random.seed(config['exp_params']['manual_seed'])

    # Load model
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model, config['exp_params'])
    
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    experiment.load_state_dict(checkpoint['state_dict'])
    model = experiment.model
    model.eval()
    model.to(args.device)

    # Load data
    data = VAEDataset(**config["data_params"])
    data.setup()
    val_dataloader = data.val_dataloader()
    
    model_name = config['model_params']['name']
    print(f"Analyzing model: {model_name}")
    print(f"Using checkpoint: {args.checkpoint_path}")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")

    latent_vectors = []
    labels = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input, label = batch # AFHQ dataset always returns image and label
            input = input.to(args.device)
            
            if model_name == 'ConditionalVAE':
                # CVAE's forward returns [recons, input, mu, log_var]
                outputs = model.forward(input, labels=label.to(args.device))
                mu = outputs[2]
                log_var = outputs[3]
            else:
                # Other VAEs have a simpler encode method
                mu, log_var = model.encode(input)

            z = model.reparameterize(mu, log_var)
            
            latent_vectors.append(z.cpu().numpy())
            labels.append(label.cpu().numpy())
            
            if args.limit_batches > 0 and i >= args.limit_batches - 1:
                print(f"Processed {i+1} batches (limit reached).")
                break

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Extracted {len(latent_vectors)} latent vectors.")

    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, 
                random_state=config['exp_params']['manual_seed'], 
                perplexity=min(args.perplexity, len(latent_vectors) - 1),
                max_iter=1000,
                n_jobs=-1)
    tsne_results = tsne.fit_transform(latent_vectors)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df['label'] = labels
    
    # Get class names from the dataset
    try:
        class_names = val_dataloader.dataset.classes
        df['class'] = [class_names[int(l)] for l in labels]
    except Exception as e:
        print(f"Could not get class names, using numeric labels. Error: {e}")
        df['class'] = df['label']


    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    
    sns.scatterplot(
        x='tsne1', y='tsne2',
        hue='class',
        palette=sns.color_palette("hsv", len(df['class'].unique())),
        data=df,
        legend="full",
        alpha=0.7
    )
    
    plt.title(f"t-SNE of Latent Space - {model_name}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    # Save plot
    output_filename = f"results_analysis/tsne_{model_name}_{args.exp_name}.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Space Analysis with t-SNE')
    parser.add_argument('--config',  '-c',
                        dest="config_path",
                        help="path to the config file",
                        required=True)
    parser.add_argument('--checkpoint', '-ckpt',
                        dest="checkpoint_path",
                        help="path to the checkpoint file",
                        required=True)
    parser.add_argument('--name', '-n',
                        dest="exp_name",
                        help="A name for the experiment to be used in the output file name",
                        default="exp")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help="device to run the model on (cuda or cpu)")
    parser.add_argument('--limit_batches', '-lb',
                        type=int,
                        default=50,
                        help="limit the number of batches to process for speed (-1 for all)")
    parser.add_argument('--perplexity', '-p',
                        type=int,
                        default=30,
                        help="t-SNE perplexity")

    args = parser.parse_args()
    analyze(args)