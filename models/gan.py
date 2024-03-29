import torch
import torch.distributions as dist
import numpy as np
from torch import nn

class Discriminator(nn.Module):
    """
    Discriminator neural network for GAN.

    Attributes:
        model (nn.Sequential): Neural network model architecture.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(19, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Discriminator output.
        """
        output = self.model(x)
        return output

class Generator(nn.Module):
    """
    Generator neural network for GAN.

    Attributes:
        model (nn.Sequential): Neural network model architecture.
    """
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 19),
        )
    
    def forward(self, x):
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Input tensor (latent space samples).

        Returns:
            torch.Tensor: Generated samples.
        """
        output = self.model(x)
        return output

class GAN:
    """
    Generative Adversarial Network (GAN) for pathway optimization.

    Attributes:
        latent_size (int): Size of the latent space.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimizers.
        num_epochs (int): Number of training epochs.
        data (numpy.ndarray): Input data for training.
        discriminator (Discriminator): Discriminator neural network.
        generator (Generator): Generator neural network.
    """
    def __init__(self, latent_size, batch_size, lr, num_epochs, data):
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.data = data
        self.discriminator = Discriminator().float()
        self.generator = Generator(self.latent_size).float()

    def train(self):
        """
        Train the GAN model.
        """
        train_data_length = len(self.data)
        train_data = torch.from_numpy(self.data)
        train_labels = torch.zeros(train_data_length)
        train_set = [
            (train_data[i], train_labels[i]) for i in range(train_data_length)
        ]
        batch_size = self.batch_size
        latent_size = self.latent_size
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        # Hyperparameters
        lr = self.lr
        num_epochs = self.num_epochs
        loss_function = nn.BCELoss()

        # Optimizers
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, weight_decay=0.001)

        # Train loop
        for epoch in range(num_epochs):
            for n, (real_samples, _) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples_labels = torch.ones((batch_size, 1))
                latent_space_samples = torch.randn((batch_size, latent_size))
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )
                
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples.float())
                loss_discriminator = loss_function(output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, latent_size))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
                loss_generator.backward()
                optimizer_generator.step()

    def generate(self, num_samples):
        """
        Generate synthetic samples using the trained generator.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        latent_space_samples = torch.randn((num_samples, self.latent_size))
        self.generator.zero_grad()
        return self.generator(latent_space_samples)
