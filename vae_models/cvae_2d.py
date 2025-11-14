import torch
from vae_models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE2D(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size:int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE2D, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.hidden_dims = hidden_dims
        self.final_img_size = img_size // (2 ** len(self.hidden_dims))

        # To account for the extra label channel
        encoder_in_channels = in_channels + num_classes
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            encoder_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], self.latent_dim, kernel_size=3, padding=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], self.latent_dim, kernel_size=3, padding=1)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Conv2d(self.latent_dim, hidden_dims[-1], kernel_size=3, padding=1)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels']
        y_one_hot = F.one_hot(y, self.num_classes).float()

        # Reshape y_one_hot to be broadcastable
        y_one_hot_map = y_one_hot.view(-1, self.num_classes, 1, 1)
        y_one_hot_map = y_one_hot_map.expand(-1, -1, input.shape[2], input.shape[3])

        x = torch.cat([input, y_one_hot_map.to(input.device)], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = [1, 2, 3]), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels']
        # y_one_hot = F.one_hot(y, self.num_classes).float().to(current_device) # Not used in decode

        z = torch.randn(num_samples,
                        self.latent_dim,
                        self.final_img_size,
                        self.final_img_size)

        z = z.to(current_device)

        # The original implementation of decode does not use the label.
        # Following that pattern here.
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]
