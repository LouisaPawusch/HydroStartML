import torch

class early_stopper:

    def __init__(self, wait=10, acceptable_diff=1):
        self.wait = wait
        self.acceptable_diff = acceptable_diff
        self.n = 0
        self.min_vali_loss = float('inf')

    def early_stop(self, vali_loss):
        if vali_loss < self.min_vali_loss:
            self.min_vali_loss = vali_loss
            self.n = 0
        elif vali_loss > (self.min_vali_loss + self.acceptable_diff):
            self.n += 1
            if self.n >= self.wait:
                return True
        return False
    
class weight_init:

    def __init__(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_weights(self, model):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        for m in model.modules():

            if isinstance(m, (torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
            if isinstance(m, (torch.nn.Conv3d)):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)