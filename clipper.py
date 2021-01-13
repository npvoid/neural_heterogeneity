import torch
import numpy as np


class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'alpha'):
            # module.alpha.data = torch.clamp(module.alpha.data, 1/(np.e**(1/4)), 0.995)  # Clip tau to [4*time_step, ]
            module.alpha.data = torch.clamp(module.alpha.data, 2/np.e, 0.995)
        if hasattr(module, 'beta'):
            module.beta.data = torch.clamp(module.beta.data, 2/np.e, 0.995)
        if hasattr(module, 'th'):
            module.th.data = torch.clamp(module.th.data, 0.5, 1.5)
