import pickle
import numpy as np
import os
from pathlib import WindowsPath, PosixPath, Path

# obscene hack to load this data
def hacked_new(cls, *args, **kwargs):
    if cls is Path:
        cls = WindowsPath if os.name == 'nt' else PosixPath
    self = cls._from_parts(args, init=False)
    # this line causes a problem
#     if not self._flavour.is_supported:
#         raise NotImplementedError("cannot instantiate %r on your system"
#                                   % (cls.__name__,))
    self._init()
    return self

def get_taum(x):
    taum = []
    for IV in list(x.IV):
        cur_taum = []
        for d in IV.values():
            if 'taum' in d and not np.isnan(d['taum']):
                cur_taum.append(d['taum']*1e3)
        if len(cur_taum):
            taum.append(np.mean(cur_taum))
    taum = np.array(taum)
    return taum

old_Path_new = Path.__new__
Path.__new__ = hacked_new # replace problematic method temporarily
x = pickle.load(open('LDA_data.pkl', 'rb'))
Path.__new__ = old_Path_new # restore old

taum = get_taum(x)
print(f'Found {len(taum)} cells with taum')
np.savetxt('manis_taum.txt', taum)
