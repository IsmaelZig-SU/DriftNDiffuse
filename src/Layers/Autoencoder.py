import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, args, model_eval = False):
        super(CNN, self).__init__()
        
        self.args = args
        print(f"AE_Model: CAE network, {self.args['num_obs']} observables")
    
 
        if not model_eval:

            self.input_size  = self.args["statedim"]
            self.latent_dim = self.args["num_obs"]
            self.activation = nn.SiLU()

        self.encoder = nn.Sequential(

            nn.Conv2d(self.input_size[0], 8, kernel_size=3, stride=2, padding=1),   # 2×64×64 → 4×32×32
            self.activation,

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),   # 4×32×32 → 8×16×16
            self.activation,

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8×16×16 → 16×8×8  
            self.activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16×8×8 → 32×4×4

        )

        self.flatten_dim = 64 * 4 * 4

        self.fc_enc = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            self.activation,    # 64 → 32 4x4->8x8
 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            self.activation,    # 32 → 16  8x8->16x16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            self.activation,    # 16 → 8 16x16->32x32

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, self.input_size[0], kernel_size=3, stride=1, padding=1),
            # 8 → 2 32x32->64x64

        )


    def encode(self, x):

        h = self.encoder(x)
        h = h.view(-1, self.flatten_dim)
        z = self.fc_enc(h)

        return z

    def recover(self, z):

        h = self.fc_dec(z)
        h = h.view(-1, 64, 4, 4)

        return self.decoder(h)

    def forward(self, x):

        z  = self.encode(x)
        x_recon = self.recover(z)

        return z,  x_recon


class CVAE(nn.Module):

    def __init__(self, args, model_eval = False):
        super(CVAE, self).__init__()
        
        self.args = args
        print(f"AE_Model: CVAE network, {self.args['num_obs']} observables, KLD regularization (beta) : {self.args['beta']}")
    
 
        if not model_eval:

            self.input_size  = self.args["statedim"]
            self.latent_dim = self.args["num_obs"]
            self.activation = nn.SiLU()

        self.encoder = nn.Sequential(

            nn.Conv2d(self.input_size[0], 8, kernel_size=3, stride=2, padding=1),   # 2×64×64 → 4×32×32
            self.activation,

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),   # 4×32×32 → 8×16×16
            self.activation,

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8×16×16 → 16×8×8  
            self.activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16×8×8 → 32×4×4

        )

        self.flatten_dim = 64 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            self.activation,    # 64 → 32 4x4 -> 8x8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            self.activation,    # 32 → 16  8x8->16x16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            self.activation,    # 16 → 8 16x16->32x32

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, self.input_size[0], kernel_size=3, stride=1, padding=1),
            # 8 → 2 32x32->64x64

        )


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def encode(self, x):

        h = self.encoder(x)
        h = h.view(-1, self.flatten_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def recover(self, z):

        h = self.fc_dec(z)
        h = h.view(-1, 64, 4, 4)

        return self.decoder(h)

    def forward(self, x):

        z, mu, logvar = self.encode(x)
        x_recon = self.recover(z)

        return z,  x_recon, mu, logvar


class AE(nn.Module):

    def __init__(self, args, model_eval = False):
        super(AE, self).__init__()
        
        self.args = args
        print(f"Dense Autoencoder, {self.args['num_obs']} observables")
    
 
        if not model_eval:

            self.input_size  = self.args["statedim"]
            self.latent_dim = self.args["num_obs"]
            self.activation = nn.SiLU()
 
        if not model_eval:
 
            #encoder layers
            self.e_fc1 = nn.Linear(self.input_size[0], 256)
            self.e_fc2 = nn.Linear(256, 128)
            self.e_fc3 = nn.Linear(128, self.latent_dim)
 
            #decoder layers
            self.d_fc1 = nn.Linear(self.latent_dim, 128)
            self.d_fc2 = nn.Linear(128, 256)
            self.d_fc3 = nn.Linear(256, self.input_size[0])
 

    def encoder(self, x):

        x = self.activation(self.e_fc1(x))
        x = self.activation(self.e_fc2(x))
        x = self.e_fc3(x)

        
        return x
    
    def decoder(self, x):
 
        #non linear encoder
        x = self.activation(self.d_fc1(x))
        x = self.activation(self.d_fc2(x))
        x = self.d_fc3(x)

        return x
 
    def forward(self, Phi_n):
        x_n       = self.encoder(Phi_n)
        Phi_n_hat = self.decoder(x_n)
 
        return x_n, Phi_n_hat
 
    def encode(self, Phi_n):

        x_n = self.encoder(Phi_n)
        return x_n
 

    def recover(self, x_n):
        
        Phi_n_hat = self.decoder(x_n)
        return Phi_n_hat
 