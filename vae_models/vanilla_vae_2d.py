import torch
from vae_models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from modules.distributions import DiagonalGaussianDistribution


class VanillaVAE2D(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 kld_anneal_end_step: int = 0,
                 **kwargs) -> None:
        super(VanillaVAE2D, self).__init__()

        self.latent_dim = latent_dim
        self.kld_anneal_end_step = kld_anneal_end_step

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.final_img_size = img_size // (2 ** len(self.hidden_dims))


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

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

    def encode(self, input: Tensor) -> "DiagonalGaussianDistribution":
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (DiagonalGaussianDistribution) The posterior distribution.
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        # Combine mu and log_var to match the parameters format for DiagonalGaussianDistribution
        parameters = torch.cat([mu, log_var], dim=1)
        posterior = DiagonalGaussianDistribution(parameters)

        return posterior

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H' x W']
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        posterior = self.encode(input)
        z = posterior.sample()
        return  [self.decode(z), input, posterior]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        posterior = args[2]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        # KL Annealing
        if self.kld_anneal_end_step > 0:
            global_step = kwargs.get('global_step', 0)
            kld_weight *= min(1.0, global_step / self.kld_anneal_end_step)

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(posterior.kl())

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 
                'Reconstruction_Loss':recons_loss.detach(), 
                'KLD':-kld_loss.detach(), 
                'kld_weight': torch.tensor(kld_weight)}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim,
                        self.final_img_size,
                        self.final_img_size)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
