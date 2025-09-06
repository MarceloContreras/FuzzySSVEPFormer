import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn.manifold import TSNE

# figure size in inches
rcParams["figure.figsize"] = 20.7, 8.27

for time in [1.0, 10.0, 100.0, 1000.0, 100000.0]:
    tsne = TSNE(n_components=2, perplexity=40)
    X, Y = [], []
    for sub in range(1, 11):
        X_tsne = np.load(
            f"/home/nodelab/Documents/Marcelo/Thesis/DCA/representations/Encoder_tsne/Nakanishi_noise/X_ssvepformer_sub{sub}_t1.0_n{time}.npy"
        )
        Y_tsne = np.load(
            f"/home/nodelab/Documents/Marcelo/Thesis/DCA/representations/Encoder_tsne/Nakanishi_noise/Y_ssvepformer_sub{sub}_t1.0_n{time}.npy"
        )
        X.append(X_tsne)
        Y.append(Y_tsne)

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    Y = Y.reshape(Y.shape[0] * Y.shape[1])

    X = tsne.fit_transform(X)

    tsne_data = np.vstack((X.T, Y)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    sn.FacetGrid(tsne_df, hue="label", palette="tab10", height=10).map(
        plt.scatter, "Dim_1", "Dim_2"
    ).add_legend()
    plt.savefig(f"t_sne/ssvepformer_1.0s_n{time}.png")
    plt.close()
