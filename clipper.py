import torch
import numpy as np


class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'alpha'):
            # module.alpha.data = torch.clamp(module.alpha.data, 1/(np.e**(1/4)), 0.995)  # Clip tau to [4*time_step, ]
            module.alpha.data.clamp_(2/np.e, 0.995)
        if hasattr(module, 'beta'):
            module.beta.data.clamp_(2/np.e, 0.995)
        if hasattr(module, 'th'):
            module.th.data.clamp_(0.5, 1.5)
