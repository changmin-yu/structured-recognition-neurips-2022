import matplotlib.pyplot as plt
import numpy as np
import io
import PIL.Image
from torchvision.transforms import ToTensor

colours = ('b', 'g', 'r', 'c', 'm', 'navy', 'lime', 'y', 'k', 'pink', 'orange', 'magenta', 'firebrick', 'olive',
          'aqua', 'sienna', 'khaki', 'teal', 'darkviolet', 'darkseagreen')
markers = [',', '+', '.', '*', 'x', 'd', 'v', '>', '<', '^', 'o']

ps_train = 5
ps_test = 30
cluster_size = 100
linewidth_cluster_ellipsis = 2.

def plot_clustered_data(y_train, y_test, clusters, ax=None):
    min = clusters.min()
    max = clusters.max()
    if ax is None:
        ax = plt
    if y_train is not None:
        ax.scatter(y_train[:, 0], y_train[:, 1], color='lightgray', s=ps_train)
    for i in range(min, max+1):
        ax.scatter(y_test[:, 0][clusters==i], y_test[:, 1][clusters==i], marker=markers[i%len(markers)], 
                   s=ps_test, color=colours[i%len(colours)])
    
def plot_components(mu_k, sigma_k, pi, ax):
    K, _ = mu_k.shape
    if ax is None:
        ax=plt
    ax.scatter(mu_k[:, 0], mu_k[:, 1], color=colours[:K], marker='D', s=cluster_size)
    
    # ellipse
    if mu_k.shape[1] == 2:
        for k, weight in enumerate(pi):
            if weight > 0.3 / K:
                t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
                circle = np.vstack((np.sin(t), np.cos(t)))
                ellipse = 2 * np.dot(np.linalg.cholesky(sigma_k[k, :]), circle) + mu_k[k, :, None].detach().numpy()
                ax.plot(ellipse[0], ellipse[1], alpha=weight.item(), linestyle='-', linewidth=linewidth_cluster_ellipsis, 
                        color=colours[k%len(colours)])

def plot_clusters(X, mu_k, sigma_k, r_nk, pi, ax=None, title='Clusters'):
    if ax is None:
        f, ax = plt.subplots()
    clusters = r_nk.argmax(axis=1)
    plot_clustered_data(None, X, clusters, ax=ax)
    if title is not None:
        ax.set_title(title)
    plot_components(mu_k, sigma_k, pi, ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image