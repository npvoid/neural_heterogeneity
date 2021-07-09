import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from allensdk.core.cell_types_cache import CellTypesCache

# We pick three plots. First plot is the Mouse CN data from Paul Manis, because this the largest collection of auditory neurons. Good fit with Gamma, OK fit with Log Normal.
# Next two plots are from Allen data because it is cleaner than neuroelectro in terms of dividing into excitatory/inhibitory subtypes which seem to have similar distributions but with different parameters, so ought to be separated.
# We then show one mouse and one human region to have two different species, and pick the largest region/neuron type combination for each. In both cases, this is spiny (putatively excitatory) neurons. the mouse region is "Primary visual area, layer 4" and the human region is "middle temporal gyrus".

# Fitting and testing fits
min_per_bin = 20 # how many neurons we want in each bin for the Chi2 test
distributions = { # dist, ddof, initargs, kwds
    'Gamma': (stats.gamma, 2, (7.0,), dict(floc=0.0), '--'), # floc=0 forces the loc parameter to be 0 for scipy.stats
    'Lognormal': (stats.lognorm, 2, (), dict(floc=0.0), ':'),
    }

# Load Allen data
ctc = CellTypesCache()
cells = ctc.get_cells()
ephys = ctc.get_ephys_features()

taus = {}
for ephy in ephys:
    taus[ephy['specimen_id']] = ephy['tau']

# Load Manis data
tau_manis = np.loadtxt('manis_taum.txt')

show_datasets = {
    'Manis: Mouse CN': tau_manis,
    'Allen: Mouse V1/L4 (exc)': np.array([taus[cell["id"]] for cell in cells if cell['dendrite_type'] =='spiny' and cell['structure_area_abbrev'] == "VISp" and cell['structure_layer_name'] == "4" and cell['id'] in taus.keys()]),
    'Allen: Human MTG (exc)': np.array([taus[cell["id"]] for cell in cells if cell['dendrite_type'] =='spiny' and cell['structure_area_abbrev'] == "MTG" and cell['id'] in taus.keys()]),
    }

# Initialise figure handle
fig = plt.figure(figsize=(12, 4), dpi=100, constrained_layout=True)
spec = gridspec.GridSpec(nrows=1, ncols=len(show_datasets), figure=fig)

for i, (name, y) in enumerate(show_datasets.items()):
    ax = fig.add_subplot(spec[i])
    _, bins, _ = plt.hist(y, bins=np.linspace(0, 80, 80), density=True, alpha=0.7)
    #plt.axvline(np.mean(y), ls='--', c='k', label=f'Mean {np.mean(y):.1f} ms')
    for distname, (dist, ddof, initargs, kwds, linestyle) in distributions.items():
        params = dist.fit(y, *initargs, **kwds)
        mledist = dist(*params)
        print(distname, params)
        # statistical analysis (chi-square)
        y_sorted = np.sort(y)
        num_bins = len(y)//min_per_bin
        chi_bins = y_sorted[np.array(np.linspace(0, 1, num_bins)*(len(y)-1), dtype=int)]
        chi_bins[0] = 0
        chi_bins[-1] = 1e10 # effectively inf
        counts, _ = np.histogram(y, chi_bins)
        expected_counts = np.diff(mledist.cdf(chi_bins))*len(y)
        chi2, pvalue = stats.chisquare(counts, expected_counts, ddof=ddof)
        print(distname, chi2, pvalue)
        plt.plot(bins, mledist.pdf(bins), color='k', linestyle=linestyle, label=distname)
        # plt.plot(bins, mledist.pdf(bins), lw=2, label=f'{distname}, p={pvalue:.3f}')
    plt.title(name, fontsize=14)
    #ax.set_frame_on(False)
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    # if i != 2:
    #     ax.set_xticks([])

    plt.xlabel(r'Membrane time constant (ms)', fontsize=12)
    plt.legend(loc='best', fontsize="large")

plt.savefig('experiment_fit.pdf', format='pdf')

