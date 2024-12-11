import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

class NetworkTrafficGAN:
    def __init__(self, data_path, selected_features=None):
        self.data_path = data_path
        self.selected_features = selected_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data_path)
        data = df[self.selected_features].values
        
        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = self.scaler.fit_transform(data)
        
        # Convert to PyTorch tensors
        tensor_data = torch.FloatTensor(scaled_data)
        dataset = TensorDataset(tensor_data)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return len(self.selected_features)
    
    def train(self, latent_dim=100, epochs=500):
        input_dim = self.prepare_data()
        
        # Initialize networks
        self.generator = Generator(latent_dim, input_dim).to(self.device)
        self.discriminator = Discriminator(input_dim).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            print(f"epcoh : {epoch}")
            d_losses, g_losses = [], []
            
            for real_data in self.dataloader:
                batch_size = real_data[0].size(0)
                real_data = real_data[0].to(self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, 1).to(self.device)
                label_fake = torch.zeros(batch_size, 1).to(self.device)
                
                output_real = self.discriminator(real_data)
                d_loss_real = criterion(output_real, label_real)
                
                z = torch.randn(batch_size, latent_dim).to(self.device)
                fake_data = self.generator(z)
                output_fake = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_data)
                g_loss = criterion(output_fake, label_real)
                
                g_loss.backward()
                g_optimizer.step()
                
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
            
            print(f'Epoch [{epoch}], d_loss: {np.mean(d_losses):.4f}, g_loss: {np.mean(g_losses):.4f}')
    
    def generate_samples(self, n_samples=100, latent_dim=100):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim).to(self.device)
            generated_data = self.generator(z).cpu().numpy()
            
        # Inverse transform the generated data
        generated_data = self.scaler.inverse_transform(generated_data)
        
        # Convert to DataFrame
        generated_df = pd.DataFrame(generated_data, columns=self.selected_features)
        return generated_df

# Usage example
def main():
    # Initialize and train the GAN
    gan = NetworkTrafficGAN('./data/ALLFLOWMETER_HIKARI2021.csv')
    gan.train(epochs=1000)
    
    # Generate synthetic samples
    synthetic_data = gan.generate_samples(n_samples=1000)
    
    # Save synthetic data
    synthetic_data.to_csv('synthetic_network_traffic.csv', index=False)
    
    print("Synthetic data generation complete!")

if __name__ == "__main__":
    main()