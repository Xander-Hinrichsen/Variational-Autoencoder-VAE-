import torch
import torch.nn as nn


##kl loss + recon loss (mse of input,output)
class ELBOLoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.beta = beta
    def forward(self, xb, z, logvar, mean, std, xhat):
        #creating a distribution for each dimension of z based on encoded means and variations
        #creating a priori distribution that we want to move closer to through optimization
        prior_dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        z_dist = torch.distributions.Normal(mean, std)
        
        ##find the log of probability that z was sampled from z's distribution and the priori
        log_probp = prior_dist.log_prob(z)
        log_probz = z_dist.log_prob(z)
        
        ##putting dim as -1 to ensure that these log probabilities are only 1 dimensional
        kl_loss = torch.mean(torch.sum(log_probz - log_probp, dim=1,keepdims=True), dim=0)
        mse_loss = self.mse(xb, xhat) / xb.shape[0]
        #return ((beta) * kl_loss) + ((1-beta) * mse_loss)
        return mse_loss + (self.beta*kl_loss)