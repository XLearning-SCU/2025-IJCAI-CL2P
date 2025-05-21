import torch
from torch import nn
import math
from models.consistency_models import ViewConsistentAE
from models.specificity_models import ViewSpecificAE
from models.mi_estimators import CLUBSample




class CL2P(nn.Module):

    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        self.args = args
        self.views = self.args.dataset.views
        self.device = device
        self.c_dim = self.args.consistency.c_dim
        self.s_dim = self.args.specificity.s_dim
        
        # consistency encoder.
        self.consis_enc = ViewConsistentAE(dataset=self.args.dataset.name,
                                        basic_hidden_dim=self.args.consistency.basic_hidden_dim,
                                        c_dim=self.args.consistency.c_dim,
                                        continuous=self.args.consistency.continuous,
                                        in_channel=self.args.consistency.in_channel,
                                        num_res_blocks=self.args.consistency.num_res_blocks,
                                        ch_mult=self.args.consistency.ch_mult,
                                        block_size=self.args.consistency.block_size,
                                        temperature=self.args.consistency.temperature,
                                        latent_ch=self.args.consistency.latent_ch,
                                        kld_weight=self.args.consistency.kld_weight,
                                        views=self.views,
                                        categorical_dim=self.args.dataset.class_num, 
                                        fusion=self.args.fusion.type,
                                        anneal=self.args.consistency.anneal
                                        )
            
            

        # create view-specific encoder.
        for i in range(self.views):
            self.__setattr__(f"venc_{i+1}", ViewSpecificAE(
                                                        dataset=self.args.dataset.name,
                                                        c_dim=self.args.consistency.c_dim, 
                                                        c_enable=self.args.consistency.enable,
                                                        s_dim=self.s_dim, 
                                                        latent_ch=self.args.specificity.latent_ch, 
                                                        num_res_blocks=self.args.specificity.num_res_blocks,
                                                        block_size=self.args.specificity.block_size,
                                                        channels=self.args.consistency.in_channel, 
                                                        basic_hidden_dim=self.args.specificity.basic_hidden_dim,
                                                        ch_mult=self.args.specificity.ch_mult,
                                                        init_method=self.args.backbone.init_method,
                                                        kld_weight=self.args.specificity.kld_weight,
                                                        
                                                        number_components=self.args.pseudoinputs.number,
                                                        input_size=self.args.pseudoinputs.input_size,
                                                        use_training_data_init=self.args.pseudoinputs.training_data_init,
                                                        pseudoinputs_mean=self.args.pseudoinputs.mean,
                                                        pseudoinputs_std=self.args.pseudoinputs.std,
                                                        nonlinear=self.args.pseudoinputs.nonlinear,
                                                        GTM=self.args.pseudoinputs.GTM,
                                                        grid_size=self.args.pseudoinputs.grid_size,
                                                        use_ddp = self.args.train.use_ddp,
                                                        device=self.device))
        
        
        for i in range(self.views):
            self.__setattr__(f"mi_est_{i+1}", CLUBSample(self.c_dim, self.s_dim, hidden_size=self.args.disent.hidden_size))
        
        
    def get_disentangling_params(self, vid):
        params = [self.__getattr__(f"venc_{vid+1}").get_encoder_params(),
                  self.__getattr__(f"mi_est_{vid+1}").parameters()]
        return params

    def get_reconstruction_params(self, vid):
        params = self.__getattr__(f"venc_{vid+1}").parameters()
        return params
    
    def get_vsepcific_params(self, vid):
        params = [self.__getattr__(f"venc_{vid+1}").parameters(),
                  self.__getattr__(f"mi_est_{vid+1}").parameters()]
        return params
    
    
    def forward(self, Xs, recon=False):
        
        C = self.consis_enc.consistency_features(Xs, recon)

        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out, _, _, _ = venc(Xs[i], C, recon)
            outs.append(out)
        return outs
        

    def init_pseudoinputs(self, samples, fusion, device):

        for i in range(self.views):
            view_samples = samples[i]
            self.__getattr__(f"venc_{i+1}").init_pseudoinputs(view_samples, fusion, device)
            

    @torch.no_grad()
    def sampling(self, samples_nums, device='cpu'):
        """
        samples_num: e
        """
        C = self.consis_enc.sampling(samples_nums // 2, device, return_code=True)
        outs = []
        for b in range(samples_nums):
            C = torch.cat([C[b, :].unsqueeze(0)]*samples_nums, dim=0)
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                out = venc.sample(samples_nums, C)
                outs.append(out)
        return torch.cat(outs)

    @torch.no_grad()
    def pseudo_sampling(self, sample_nums, device='cpu'):
        
        random_indices = torch.randperm(self.args.pseudoinputs.number)[:sample_nums]
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out = venc.pseudo_sampling(sample_nums, random_indices)
            outs.append(out)
        
        return torch.cat(outs)

    def get_loss(self, Xs, epoch):

        # consistent loss
        consistent_loss, return_details, C = self.consis_enc.get_loss(Xs, epoch)

        # specific_losses
        specific_losses = 0.
        disent_losses = 0.
        for i in range(self.views):
            
            venc = self.__getattr__(f"venc_{i+1}")
            recons_loss, kld_loss, S = venc.get_loss(Xs[i], y=C)
            # return_details[f'v{i+1}-recon-loss'] = recons_loss.item()
            # return_details[f'v{i+1}-kld-loss'] = kld_loss.item()
            
            specific_loss = recons_loss + kld_loss
            return_details[f'v{i+1}-specific-loss'] = specific_loss.item()
            
            mi_est = self.__getattr__(f"mi_est_{i+1}")
            mi_lb = mi_est.learning_loss(S, C)
            disent_loss = self.args.disent.lam * mi_lb
            return_details[f'v{i+1}-disent-loss'] = disent_loss.item()
            
            specific_losses += specific_loss
            disent_losses += disent_loss

        # curriculum learning
        # Lambda = 1 - ((epoch+1)/self.args.train.epochs) ** self.args.train.alpha
        lambda_ = self._gen_lambda(epoch)
        joint_loss = lambda_ * consistent_loss + (1 - lambda_) * (specific_losses + disent_losses)
        return_details[f'lambda']     = lambda_
        return_details[f'joint-loss'] = joint_loss.item()
        
        
        return joint_loss, return_details
    

    def __fusion(self, Xs, ftype='C'):

            
        consis_features = self.consis_enc.consistency_features(Xs)

        vspecific_features = []
        if self.args.specificity.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)

        if ftype == 'C':
            features = consis_features
        elif ftype == "V":
            features = vspecific_features[self.args.specificity.best_view]
        elif ftype == "CV":
            best_view_features = vspecific_features[self.args.specificity.best_view]
            features = torch.cat([consis_features, best_view_features], dim=-1)
        else:
            raise ValueError("Less than one kind information available.")

        return features


    def generate(self, z):
        outs = []
        for i in range(self.views):
            venc = self.__getattr__(f"venc_{i+1}")
            out = venc.decode(z)
            outs.append(out)
        return torch.cat(outs)
    
    @torch.no_grad()
    def all_features(self, Xs):
        
        batch = Xs[0].shape[0]
            
        C = self.consis_enc.consistency_features(Xs)

        if self.args.specificity.enable:
            vspecific_features = []
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)

            all_V = torch.cat(vspecific_features, dim=-1)
        else:
            V = torch.zeros(batch, self.s_dim).to(self.device)
            all_V = V
        return C, all_V

    @torch.no_grad()
    def consistency_features(self, Xs):
        return self.__fusion(Xs, ftype='C')

    @torch.no_grad()
    def vspecific_features(self, Xs, single=False):
        
        vspecific_features = []
        if self.args.specificity.enable:
            for i in range(self.views):
                venc = self.__getattr__(f"venc_{i+1}")
                feature = venc.latent(Xs[i])
                vspecific_features.append(feature)
        if single:
            return vspecific_features[self.args.specificity.best_view]
        return vspecific_features


    def _gen_lambda(self, epoch: int) -> float:
        """
        Generate the adaptive parameter according to the current epoch.
        
        Parameters
        ----------
        epoch : int
            The current epoch.
            
        Returns
        -------
        float
            The trade-off parameter.
        """
        T_T_max = (epoch + 1) / self.args.train.epochs
        
        if   self.args.train.scheduler == "linear":
            return 1.0 - T_T_max
        elif self.args.train.scheduler == "exponential":
            return math.exp(-2 * math.pi * T_T_max)
        elif self.args.train.scheduler == "cosine":
            return math.cos(T_T_max * math.pi) / 2.0 + 0.5 
        elif self.args.train.scheduler == "power":
            return 1.0 - T_T_max ** self.args.train.alpha
        else:
            assert False, f"Scheduler type [{self.args.train.scheduler}] not in pre-defined."
            
            