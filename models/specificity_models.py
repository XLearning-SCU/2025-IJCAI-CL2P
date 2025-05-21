import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.distributed as dist
import torch.utils.data
from torch.autograd import Variable
from models.autoencoder import Encoder, Decoder


class ViewSpecificAE(nn.Module):
    
    def __init__(self, 
                dataset=None,
                c_dim=10, 
                c_enable=True,
                s_dim=15, 
                latent_ch=10, 
                num_res_blocks=3,
                block_size=8,
                channels=1, 
                basic_hidden_dim=16,
                ch_mult=[1,2,4,8],
                kld_weight=0.00025,
                init_method='kaiming',
                number_components=100,
                input_size=[3, 64, 64],
                use_training_data_init=False,
                pseudoinputs_mean=-0.05,
                pseudoinputs_std=0.01,
                nonlinear=2,
                GTM=False,
                grid_size=2.5,
                use_ddp = False,
                device='cpu',) -> None:
        super().__init__()
        
        # view-specific id.
        self.latent_ch = latent_ch
        self.device = device
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.use_ddp = use_ddp
        self.block_size = block_size
        self.basic_hidden_dim = basic_hidden_dim
        self.s_dim = s_dim
        self.c_dim = c_dim
        self.c_enable = c_enable
        self.input_channel = channels
        self.kld_weight = kld_weight
        self.build_encoder_and_decoder()

        self.number_components=number_components
        self.input_size=input_size
        self.use_training_data_init=use_training_data_init
        self.pseudoinputs_mean= pseudoinputs_mean
        self.pseudoinputs_std=pseudoinputs_std
        self.nonlinear = nonlinear
        self.GTM=GTM
        self.grid_size=grid_size
        self.device=device
        
        self.recons_criterion = nn.MSELoss(reduction='sum')
        # self.recons_criterion = nn.MSELoss()
        # self.apply(self.weights_init(init_type=init_method))
        
        # vampprior
        self.LP = False if number_components == 0 else True
        if self.LP:
            self.add_pseudoinputs()
        
        
    def add_pseudoinputs(self):

        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0) 
        self.means = NonLinear(self.number_components, np.prod(self.input_size), bias=False, activation=nonlinearity)
        normal_init(self.means.linear, self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = torch.eye(self.number_components, self.number_components, requires_grad=False).to(self.device)
                                
                                    
    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun
                
        
    def build_encoder_and_decoder(self):
        self._encoder = Encoder(hidden_dim=self.basic_hidden_dim, 
                                in_channels=self.input_channel, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult, 
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        self._decoder = Decoder(hidden_dim=self.basic_hidden_dim, 
                                out_channels=self.input_channel, 
                                in_channels=self.latent_ch, 
                                z_channels=self.latent_ch, 
                                ch_mult=self.ch_mult,
                                num_res_blocks=self.num_res_blocks, 
                                resolution=1, 
                                use_attn=False, 
                                attn_resolutions=None,
                                double_z=False)
        
        self.to_dist_layer = nn.Linear(self.latent_ch * (self.block_size **2), self.s_dim*2)
        if self.c_enable:
            self.to_decoder_input = nn.Linear(self.s_dim+self.c_dim, self.latent_ch * (self.block_size **2))
        else:
            self.to_decoder_input = nn.Linear(self.s_dim, self.latent_ch * (self.block_size **2))
            
    
    def init_pseudoinputs(self, samples, fusion, device):
        
        samples = torch.stack(samples)        
        n = samples.shape[0]
        samples = samples.view(n, -1)  
        samples = samples.to(device)
        
        self.means.linear.weight.data = samples.T

    def get_encoder_params(self):
        return self._encoder.parameters()
    
    def latent(self, x):
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1)
        z = self.to_dist_layer(latent)
        mu, logvar = torch.split(z, self.s_dim, dim=1)
        # z = self.reparameterize(mu, logvar)
        # return z
        return mu
        
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        latent = self._encoder(x)
        latent = torch.flatten(latent, start_dim=1) 
        z = self.to_dist_layer(latent)
        mu, logvar = torch.split(z, self.s_dim, dim=1)

        return [mu, logvar]

    def decode(self, z):
        z = self.to_decoder_input(z)
        z = z.view(-1, self.latent_ch, self.block_size, self.block_size)
        result = self._decoder(z)
        return result

    def reparameterize(self, mu, logvar):
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


    def forward(self, x, y, recon=False):
        
        mu, logvar = self.encode(x)
        
        s = self.cont_reparameterize(mu, logvar, recon)
        z = torch.cat([s, y], dim=1)
            
        return self.decode(z), s, mu, logvar
    
    
    def get_loss(self, x, y):
        
        out, s, mu, logvar = self(x, y)
        # z = self.cont_reparameterize(mu, logvar)
        if self.LP:
            log_p_z = self.log_p_z(s)
            log_q_z = log_Normal_diag(s, mu, logvar, dim=1)
            KL = -(log_p_z - log_q_z)
            kld_loss = self.kld_weight * torch.mean(KL)
        else:
            kld_loss = self.kld_weight * (torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0))
        
        recons_loss = self.recons_criterion(out, x)
        
        return recons_loss, kld_loss, s
    
    def log_p_z(self, z):

        # z - MB x M
        C = self.number_components
        

        X = self.means(self.idle_input).view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
        
        # calculate params
        z_p_mean, z_p_logvar = self.encode(X)  # C x M

        # expand z
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)

        a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(C)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1

        return log_prior
    
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
        
        
    def sample(self, num_samples, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param y: (Tensor) controlled labels.
        :return: (Tensor)
        """
        if self.LP:
            random_indices = torch.randperm(len(self.idle_input))[:num_samples]
            mean = self.means(self.idle_input)[random_indices].view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
            
            z_sample_gen_mean, z_sample_gen_logvar = self.encode(mean)
            z = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
        else:
            z = torch.randn(num_samples, self.s_dim).to(self.device)
        
        z = torch.cat([z, y], dim=1).to(self.device)
        samples = self.decode(z)
        return samples
    
    
    def pseudo_sampling(self, num_samples, random_indices):
        
        out = self.means(self.idle_input)[random_indices].view(-1, self.input_size[0], self.input_size[1], self.input_size[2])
        
        return out
    

def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)

class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation( h )

        return h