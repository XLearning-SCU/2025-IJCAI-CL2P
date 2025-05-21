import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist
import math
from models.autoencoder import Encoder, Decoder


class ViewConsistentAE(nn.Module):

    def __init__(self,
                dataset=None,
                basic_hidden_dim=32, 
                c_dim=64, 
                continuous=True, 
                in_channel=3,
                num_res_blocks=3, 
                ch_mult=[1, 2, 4, 8],
                block_size=8,
                latent_ch=10,
                temperature=1.0,
                kld_weight=0.00025,
                views=2,
                alpha=0.5,
                categorical_dim=10,
                fusion="moe",
                anneal=0
                ) -> None:
        """
        """
        super().__init__()

        self.c_dim = c_dim
        self.continuous = continuous
        self.in_channel = in_channel
        self.ch_mult = ch_mult
        self.block_size = block_size
        self.basic_hidden_dim = basic_hidden_dim
        self.num_res_blocks = num_res_blocks
        self.latent_ch = latent_ch
        self.anneal_rate = 0.00003
        self.min_temp = 0.5
        self.temp = temperature
        self.views = views
        self.kld_weight = kld_weight
        self.categorical_dim = categorical_dim
        self.alpha = alpha
        self.fusion = fusion
        self.anneal = anneal
        
        self._encoder = Encoder(hidden_dim=self.basic_hidden_dim, 
                                in_channels=self.in_channel, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult, 
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        
    
        self.decoders = nn.ModuleList([Decoder(hidden_dim=self.basic_hidden_dim, 
                                out_channels=self.in_channel, 
                                in_channels=self.latent_ch, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult,
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False) for _ in range(self.views)])
        
        if self.fusion == "moe":
            
            self.gating = nn.ModuleList([nn.Sequential(nn.Linear(512, 256), 
                                                    nn.ReLU(),
                                                    nn.Linear(256, 1)
                                                    ) 
                                        for _ in range(self.views)])
            
            
        if self.continuous:
            # continuous code.
            self.fc_z = nn.Linear(512 * self.views, self.c_dim*2)
            self.fc_z_single = nn.ModuleList([nn.Sequential(nn.Linear(512, self.c_dim*2)) for _ in range(self.views)])
            # self.fc_z = nn.Linear(self.latent_ch * self.block_size ** 2 * self.views, self.c_dim*2)
            self.to_decoder_input = nn.Linear(self.c_dim, self.latent_ch * self.block_size **2)
        else:
            # discrete code.
            self.fc_z = nn.Linear(self.latent_ch * self.block_size ** 2, self.c_dim * self.categorical_dim)
            self.to_decoder_input = nn.Linear(self.c_dim * self.categorical_dim, self.latent_ch * self.block_size **2)
        
        
    def forward(self, Xs):
    
        if self.continuous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs)
            z = self.disc_reparameterize(beta)    
        
        recons = self.decode(z)
        return recons, z
    
    
    def cont_reparameterize(self, mu, logvar, recon=False):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if self.training or recon:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu
    
    
    def disc_reparameterize(self, z, eps=1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # if self.training:
            # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.c_dim * self.categorical_dim)
        return s
    
        
    def encode(self, Xs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        latents = []
        
        for x in Xs:
            latent = self._encoder(x)
            latent = torch.flatten(latent, start_dim=1) # [batch_size, 512]
            latents.append(latent)
        
        if self.fusion == "concat":
            latent = torch.cat(latents, dim=-1) 
            z = self.fc_z(latent)
            mu, logvar = torch.split(z, self.c_dim, dim=1)            
        elif self.fusion == "moe":
            mu_fusion, logvar_fusion, gate_weights = [], [], []
            
            for i in range(self.views):
                
                z_view = self.fc_z_single[i](latents[i])
                mu_view, logvar_view = torch.split(z_view, self.c_dim, dim=1) # [batch, c_dim]
                gate_weights.append(self.gating[0](latents[i]))
                mu_fusion.append(mu_view)
                logvar_fusion.append(logvar_view)
            
            mu_fusion = torch.stack(mu_fusion, dim=1) # [views, batch, c_dim]
            logvar_fusion = torch.stack(logvar_fusion, dim=1)
            weights = F.softmax(torch.stack(gate_weights, dim=1), dim=1)
            
            mu = (weights * mu_fusion).sum(dim=1)  
            logvar = (weights * logvar_fusion).sum(dim=1)
                
        
        if self.continuous:
            return mu, logvar
        else:
            return z.view(-1, self.c_dim, self.categorical_dim)

    def decode(self, z):
        z = self.to_decoder_input(z)
        z = z.view(-1, self.latent_ch, self.block_size, self.block_size)
        return [dec(z) for dec in self.decoders]
        
    
    def get_loss(self, Xs, epoch):
        
        if self.continuous:
            mu, logvar = self.encode(Xs)
            kld_loss = self.con_loss(mu, logvar)
            c = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs)
            kld_loss = self.disc_loss(beta, epoch)
            c = self.disc_reparameterize(beta)
        recons = self.decode(c)
        
        
        recon_loss = 0.
        return_details = {}
        for v, (x, recon) in enumerate(zip(Xs, recons)):
            sub_loss = F.mse_loss(x, recon, reduction='sum')
            # return_details[f'v{v+1}-loss'] = sub_loss.item()
            recon_loss += sub_loss

        loss = recon_loss + kld_loss

        return_details['consistent-loss'] = loss.item()
        # return_details['recon_loss'] = recon_loss.item()
        # return_details['kld_loss'] = kld_loss.item()
        
        return loss, return_details, c
    
    def con_loss(self, mu, log_var):
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return self.kld_weight * kld_loss
        
    def disc_loss(self, Q, epoch) -> dict:
        """"
        Computes the discreate-VAE loss function.
        """
        
        B, N, K = Q.shape
        Q = Q.view(B*N, K)
        q = dist.Categorical(logits=Q)
        p = dist.Categorical(probs=torch.full((B*N, K), 1.0/K).to(Q.device)) # uniform bunch of K-class categorical distributions
        kl = dist.kl.kl_divergence(q, p).view(B, N) # kl is of shape [B*N]
        
        if epoch % 5 == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * epoch),
                                self.min_temp)
        
        return torch.mean(torch.sum(kl, dim=1))

    
    def consistency_features(self, Xs, recon=False):
        if self.continuous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar, recon)
        else:
            beta = self.encode(Xs)
            z = self.disc_reparameterize(beta)
        return z
    
    def sampling(self, samples_num, device='cpu', return_code=False):
        if self.continuous:
            z = torch.randn(samples_num, self.c_dim).to(device)
        else:    
            Q = torch.randn((samples_num, self.c_dim, self.categorical_dim))
            z = self.disc_reparameterize(Q).to(device)
        if return_code:
            return z
        else:
            samples = self.decode(z)
            return samples
            