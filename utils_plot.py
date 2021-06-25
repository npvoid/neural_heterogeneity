from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import seaborn as sns
import numpy as np
import os

from scipy import sparse

def ab2tau(ab, dt):
    return -dt/np.log(ab)


def plot_layers(layer_recs):
    for layer in layer_recs:
        plt.figure()
        dim = (3, 5) if layer[1].shape[2] > 15 and layer[1].shape[0] > 15 \
            else (np.min(layer[1].shape), 1)
        if len(layer) == 3:
            plot_voltage_traces(layer[1], spk=layer[-1], spike_height=5, dim=dim)
        else:
            plot_voltage_traces(layer[1], spike_height=5, dim=dim)
        plt.close()


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = (mem+spike_height*spk).detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0:
            a0=ax=plt.subplot(gs[i])
        else:
            ax=plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i, :, i])
        # ax.axis("off")


def plot_loss(path_results, dirName, prms):
    # Plot Loss evolution
    train_loss = prms['train_loss']
    test_loss = prms['test_loss']

    plt.figure()
    sns.set()
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss)
    plt.title(str(prms['train_ab'])+"AB_"+str(prms['het_ab'])+"HET Train Loss History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path_results, dirName, 'Train Loss Evolution.png'))
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, len(test_loss) + 1), test_loss)
    plt.title(str(prms['train_ab'])+"AB_"+str(prms['het_ab'])+"HET Test Loss History")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path_results, dirName, 'Test Loss Evolution.png'))
    plt.close()


def plot_param_dist(path_results, dirName, prms, learn_prms):
    sns.set_style('darkgrid')
    # Plot distribution of synaptic and membrane time constants
    for lparam_name, lparam in learn_prms:
        layer = lparam_name.replace('.', ' ').split()[1]  # Layer number
        pname = lparam_name.replace('.', ' ').split()[2]  # Learned Parameter name
        plt.figure()
        # Time constants distribution
        if pname == 'alpha' or pname == 'beta':
            tau = ab2tau(lparam.flatten(), prms['time_step']) / 1e-3
            print(lparam_name, np.max(lparam), np.min(tau))
            ax = sns.distplot(tau, kde=False)
            plt.xlabel('ms')
            if pname == 'alpha':
                title = "Layer {} tau_syn distribution ({}AB_{}HET)".format(layer, prms['train_ab'], prms['het_ab'])
                ax.set_title(title)
            elif pname == 'beta':
                title = "Layer {} tau_mem distribution ({}AB_{}HET)".format(layer, prms['train_ab'],
                                                                                  prms['het_ab'])
                ax.set_title(title)
        elif pname == 'th':
            ax = sns.distplot(lparam.flatten(), kde=False)
            plt.xlabel('ms')
            title = "Layer {} Threshold distribution ({}TH_{}THET)".format(layer, prms['train_th'], prms['het_th'])
            ax.set_title(title)
        # Weights distribution
        else:
            ax = sns.distplot(lparam.flatten(), kde=False)
            plt.xlabel('weight')
            title = "Layer {} weights distribution ({}AB_{}HET)".format(layer, prms['train_ab'], prms['het_ab'])
            ax.set_title(title)
        plt.savefig(os.path.join(path_results, dirName, title+'.png'))
        plt.close()


def plot_input(sample, labels):
    sample = sample.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    fig = plt.figure(figsize=(16, 4))
    idx = np.random.randint(sample.shape[0], size=3)
    for i, k in enumerate(idx):
        tmp = sparse.coo_matrix(sample[k, :, :])
        units = tmp.col
        times = tmp.row
        ax = plt.subplot(1, 3, i+1)
        ax.scatter(times, units, color="k", alpha=0.33, s=2)
        ax.set_title("Label %i" % labels[k])
        ax.axis("off")


def plot_raster(path_results, dirName, prms, layer_recs_v, mode='train', batch=0):
    subdirName = 'raster_' + mode
    for e_recs in layer_recs_v:
        epoch, layer_recs = e_recs
        for l, layer in enumerate(layer_recs):
            plt.figure()
            s = layer[-1]
            if len(layer)==2:
                plt.imshow(s[batch, :, :].detach().cpu().t(), cmap=plt.cm.YlGnBu, aspect="auto")
                plt.colorbar()
            else:
                plt.imshow(s[batch, :, :].detach().cpu().t(), cmap=plt.cm.gist_gray, aspect="auto")
            plt.xlabel("Time (ms)")
            plt.ylabel("Neuron")
            title = "Epoch {} Layer {} spikes ({}AB_{}HET)".format(epoch, l, prms['train_ab'], prms['het_ab'])
            plt.title(title)
            sns.despine()
            if not os.path.exists(os.path.join(path_results, dirName, subdirName)):
                os.mkdir(os.path.join(path_results, dirName, subdirName))
            plt.savefig(os.path.join(path_results, dirName, subdirName, title+'.png'))
            plt.close()


def show2D(timestamps, xaddr, yaddr, pol, time_step, frameRate=24, preComputeFrames=True, repeat=False, minTimeStamp=None, maxTimeStamp=None):
    fig = plt.figure(dpi=200)
    interval = 1 / (frameRate * time_step)
    xDim = xaddr.max() + 1
    yDim = yaddr.max() + 1

    if minTimeStamp is None:
        minTimeStamp = timestamps.min()
    if maxTimeStamp is None:
        maxTimeStamp = timestamps.max()
        print(maxTimeStamp)


    if preComputeFrames is True:
        minFrame = int(np.floor(minTimeStamp / interval))
        maxFrame = int(np.ceil(maxTimeStamp / interval))
        image = plt.imshow(np.zeros((yDim, xDim, 3)))
        # frames = np.zeros((maxFrame - minFrame, yDim, xDim, 3))
        frames = np.ones((maxFrame - minFrame, yDim, xDim, 3))*0.5

        # precompute frames
        for i in range(len(frames)):
            tStart = np.maximum((i + minFrame) * interval, minTimeStamp)
            tEnd = np.minimum((i + minFrame + 1) * interval, maxTimeStamp)
            timeMask = (timestamps >= tStart) & (timestamps < tEnd)
            rInd = (timeMask & (pol == 1))
            gInd = (timeMask & (pol == 0))
            bInd = (timeMask)

            # frames[i, yaddr[rInd], xaddr[rInd], 0] = 1
            # frames[i, yaddr[gInd], xaddr[gInd], 1] = 1
            # frames[i, yaddr[bInd], xaddr[bInd], 2] = 1

            frames[i, yaddr[rInd], xaddr[rInd], :] = [0, 0, 0]
            frames[i, yaddr[gInd], xaddr[gInd], :] = [1, 1, 1]

        def animate(frame):
            image.set_data(frame)
            return image

        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=42, repeat=repeat)

    else:
        minFrame = int(np.floor(minTimeStamp / interval))

        def animate(i):
            tStart = np.maximum((i + minFrame) * interval, minTimeStamp)
            tEnd = np.minimum((i + minFrame + 1) * interval, maxTimeStamp)
            frame = np.zeros((yDim, xDim, 3))
            timeMask = (timestamps >= tStart) & (timestamps < tEnd)
            rInd = (timeMask & (pol == 1))
            gInd = (timeMask & (pol == 2))
            bInd = (timeMask & (pol == 0))
            frame[yaddr[rInd], xaddr[rInd], 0] = 1
            frame[yaddr[gInd], xaddr[gInd], 1] = 1
            frame[yaddr[bInd], xaddr[bInd], 2] = 1
            plot = plt.imshow(frame)
            return plot

        anim = animation.FuncAnimation(fig, animate, interval=42, repeat=repeat)  # 42 means playback at 23.809 fps

    # # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # # installed.  The extra_args ensure that the x264 codec is used, so that
    # # the video can be embedded in html5.  You may need to adjust this for
    # # your system: for more information, see
    # # http://matplotlib.sourceforge.net/api/animation_api.html
    # if saveAnimation: anim.save('showTD_animation.mp4', fps=30)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
