import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

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
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GANBalancer:
    def __init__(self, latent_dim=32, epochs=50, batch_size=128, lr=0.0002):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.generator = None

    def balance(self, X, y):
        print(f"Using device: {self.device}")
        X_df = pd.DataFrame(X)
        y_df = pd.Series(y)
        
        # Identify minority class
        counts = y_df.value_counts()
        minority_class = counts.idxmin()
        majority_class = counts.idxmax()
        
        n_minority = counts[minority_class]
        n_majority = counts[majority_class]
        n_to_generate = n_majority - n_minority
        
        if n_to_generate <= 0:
            print("Dataset is already balanced.")
            return X, y

        print(f"Training GAN to generate {n_to_generate:,} samples of class {minority_class}...")
        
        # Prepare minority data
        X_minority = X_df[y_df == minority_class].values.copy()
        X_tensor = torch.FloatTensor(X_minority).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize networks
        input_dim = X.shape[1]
        self.generator = Generator(self.latent_dim, input_dim).to(self.device)
        discriminator = Discriminator(input_dim).to(self.device)
        
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # Training Loop
        for epoch in range(self.epochs):
            for i, (real_samples,) in enumerate(dataloader):
                batch_size = real_samples.size(0)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # --- Train Discriminator ---
                optimizer_D.zero_grad()
                
                # Real samples
                outputs = discriminator(real_samples)
                d_loss_real = criterion(outputs, real_labels)
                
                # Fake samples
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_samples = self.generator(z)
                outputs = discriminator(fake_samples.detach())
                d_loss_fake = criterion(outputs, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()
                
                # --- Train Generator ---
                optimizer_G.zero_grad()
                
                outputs = discriminator(fake_samples)
                g_loss = criterion(outputs, real_labels) # We want D to think fake are real
                
                g_loss.backward()
                optimizer_G.step()
                
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
                
        # Generation Phase
        print("Generating synthetic samples...")
        self.generator.eval()
        all_fake_samples = []
        
        # Generate in batches to avoid memory issues
        gen_batch_size = 10000
        n_batches = int(np.ceil(n_to_generate / gen_batch_size))
        
        with torch.no_grad():
            for i in range(n_batches):
                current_batch_size = min(gen_batch_size, n_to_generate - i * gen_batch_size)
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                fake_samples = self.generator(z).cpu().numpy()
                all_fake_samples.append(fake_samples)
                
        X_fake = np.vstack(all_fake_samples)
        y_fake = np.full(n_to_generate, minority_class)
        
        # Combine
        X_balanced = np.vstack([X, X_fake])
        y_balanced = np.hstack([y, y_fake])
        
        return X_balanced, y_balanced
