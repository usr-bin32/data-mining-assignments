from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def dbscan_clustering(df, columns, scaling=False, eps=0.5, min_samples=5, p=None):

    # Select data and make scaling of them
    K = df[columns].values
    if(scaling):
        scaler = MinMaxScaler(feature_range=(0, 1))
        K = scaler.fit_transform(K)

    #  Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, p=p).fit(K)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)




    # #############################################################################
    # Plot result

    # Black removed and is used for noise/outlier instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure()
    fig = plt.figure(figsize=(11., 11.))
    ax = fig.add_subplot(111, projection='3d', title='dbscan')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = K[class_member_mask & core_samples_mask]
        ax.scatter(xy[:, 2], xy[:, 1], xy[:, 0], color=tuple(col), alpha=0.8, edgecolor='k', marker='o', s=200, label = k)
        
        #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)

        xy = K[class_member_mask & ~core_samples_mask]
        ax.scatter(xy[:, 2], xy[:, 1], xy[:, 0], color=tuple(col), alpha=0.8,edgecolor='k', marker='o', s=60)


    ax.set_title('Agrupamento não supervisionado DBSCAN - Agrupamentos (clusters): %d' % n_clusters_)
    ax.set_xlabel(columns[2])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[0])
    plt.legend(loc = 'best')
    plt.show()

    pd.options.display.float_format = "{:.2f}".format


    # Impressão dos clusters
    clusters = [ K[labels == i] for i in range(n_clusters_) ]
    outliers = K[labels == -1]

    for j in range(n_clusters_):
        if(scaling):
            cluster = scaler.inverse_transform(clusters[j])
        else:
            cluster = clusters[j]
        cluster = pd.DataFrame(clusters[j], columns=columns)
        print('----------------------------------------------')
        print('Grupo', j, 'com', len(cluster), 'procedimentos')
        print(cluster.describe().round(2))




df = pd.read_csv("data/trabalho4_dados_4.csv")


dbscan_clustering(df, ['maior-eixo','arredondamento','extensao', 'perimetro', 'area', 'area-convexa', 'menor-eixo'], True, eps=0.14, min_samples=36) # , p=2