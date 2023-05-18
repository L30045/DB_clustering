import numpy as np
import scipy as sp
from scipy import optimize
from scipy.stats import multivariate_normal
import sklearn as sk
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KDTree, BallTree, NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.covariance import ledoit_wolf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import cluster
import matplotlib.pyplot as plt
import umap
import umap.plot

#%% Cluster
#%% cluster pipeline
def cluster_pipe(X,eps=0.2,min_smpl=30):
    """
    (1) Standardize and PCA data
    (2) KMean to find the northeast group which is relatively far away from others
    (3) Remove northeast group
    (4) DBSCAN with esp=0.2, min_samples=30
    (5) remove cluster with points less than 1% of total data points.
    (6) for each cluster, project to 1st PC and fit gaussian.
        (test) project to 1st PLS
    (7) fit left and right outside 2*std_main data into gaussian.
    (8) If there is a second gaussian with mean 1*std_sub away from 2*std_main, make it another cluster.
    (9) threshold to divide these twe clusters is set at
        (a) the intersection of main and sub probability dist.
        (b) 1 std_sub away from mean_sub.
    (10) TODO: fit multivariate gaussian to each cluster and cluster the rest of the outlier
    (11) Calculate the centroid of each cluster and put the rest of the outlier into the closest one.
    """
    # record group
    group_lib = []
    group_label = np.ones((X.shape[0],))*-1
    group_count = 0
    
    # Standardize and PCA data
    std_scaler = StandardScaler()
    pca = PCA(n_components=2)
    X_std = std_scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_std)

    # KMean to get northest group
    fit_kmean = cluster.KMeans(n_clusters=4) # use 4 seeds as prior knowledge
    label_kmean = fit_kmean.fit_predict(X_pca)
    center_kmean = fit_kmean.cluster_centers_

    # find northest group
    group_0_idx = label_kmean==np.argmax([np.linalg.norm(x) for x in center_kmean])
    group_0_data = X[group_0_idx,:]
    group_lib.append(group_0_data)
    group_label[group_0_idx] = group_count
    group_count+=1

    # remove northeast
    trunc_data = X_pca[~group_0_idx]
    # Standardize and PCA truncated data
    trunc_pca = std_scaler.fit_transform(trunc_data)
    trunc_pca = pca.fit_transform(trunc_pca)
    
    # DBSCAN on pca-ed truncated data
    fit_db = DBSCAN(eps=eps, min_samples=min_smpl).fit(trunc_pca)
    label_db = fit_db.labels_
    print(f"# of DBSCAN in truncated data = {len(set(label_db))}")

    # for each cluster if cluster is big enough
    for cls_i in set(label_db):
        # ignore outlier group
        if cls_i != -1:
            print(f"enter cls {cls_i}")
            # extract cluster data
            cls_data = trunc_data[label_db==cls_i]
            # ignore small cluster
            if cls_data.shape[0] <group_0_data.shape[0]:
                print(f"enter break: cls{cls_i}, size {cls_data.shape[0]}")
                break
            # project cluster data onto 1st PC
            cls_pca = PCA(n_components=1).fit_transform(cls_data)
            # calculate histogram
            n, bins= np.histogram(cls_pca, bins=100)
            # fit curve
            opt_norm,_ = optimize.curve_fit(gaussian,bins[:-1],n)
            # examine left and right
            right_idx = bins[:-1]>(opt_norm[1]+opt_norm[2]*2)
            left_idx = bins[:-1]<(opt_norm[1]-opt_norm[2]*2)
            opt_norm_right = None
            opt_norm_left = None
            try:
                opt_norm_right,_ = optimize.curve_fit(gaussian,bins[:-1][right_idx],n[right_idx])
            except:
                print(f'Right tail data is not gaussian in cluster {cls_i}.')
            try:
                opt_norm_left,_ = optimize.curve_fit(gaussian,bins[:-1][left_idx],n[left_idx])
            except:                
                print(f'Left tail data is not gaussian in cluster {cls_i}.')
            print(f"right {opt_norm_right}")
            print(f"left {opt_norm_left}")
            # create new clusters
            if opt_norm_right is None and opt_norm_left is None:
                print("enter no create")
                db_idx = np.nonzero(label_db==cls_i)[0] # back to trunc data level
                raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                group_label[raw_idx] = group_count
                group_lib.append(X[raw_idx])
                group_count+=1
            elif opt_norm_right is not None:
                print("enter right")
                # TODO: switch threshold method
                # (a)
                # pred_norm = gaussian(bins[:-1][right_idx],*opt_norm)
                # pred_r = gaussian(bins[:-1][right_idx],*opt_norm_right)
                # prob_thres = find_intersect_r(pred_r,pred_norm,bins[:-1][right_idx])
                # (b)
                prob_thres = opt_norm_right[1]-opt_norm_right[2]
                if prob_thres < opt_norm[1]+opt_norm[2]*2:
                    db_idx = np.nonzero(label_db==cls_i)[0] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
                    prob_thres = None
                if prob_thres:
                    new_cls_idx = (cls_pca >= prob_thres)
                    # get index
                    # sub cluster                
                    db_idx = np.nonzero(label_db==cls_i)[0][new_cls_idx.reshape(-1)] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
                    # main cluster
                    db_idx = np.nonzero(label_db==cls_i)[0][~new_cls_idx.reshape(-1)] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
            elif opt_norm_left is not None:
                print("enter left")
                # TODO: switch threshold method
                # (a)
                # pred_norm = gaussian(bins[:-1][left_idx],*opt_norm)
                # pred_l = gaussian(bins[:-1][left_idx],*opt_norm_left)
                # prob_thres = find_intersect_l(pred_l,pred_norm,bins[:-1][left_idx])
                # (b)
                prob_thres = opt_norm_left[1]+opt_norm_left[2]
                if prob_thres > opt_norm[1]-opt_norm[2]*2:
                    db_idx = np.nonzero(label_db==cls_i)[0] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
                    prob_thres = None
                if prob_thres:
                    new_cls_idx = (cls_pca <= prob_thres)
                    # get index
                    # sub cluster                
                    db_idx = np.nonzero(label_db==cls_i)[0][new_cls_idx.reshape(-1)] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
                    # main cluster
                    db_idx = np.nonzero(label_db==cls_i)[0][~new_cls_idx.reshape(-1)] # back to trunc data level
                    raw_idx = np.nonzero(~group_0_idx)[0][db_idx] # back to raw data level
                    group_label[raw_idx] = group_count
                    group_lib.append(X[raw_idx])
                    group_count+=1
    
    # TODO: report outlier


    return group_lib, group_label, group_count

# gaussian function
def gaussian(x, amplitude, mean, std):
    return amplitude/np.sqrt(2*np.pi)/std*np.exp(-((x-mean)/std)**2/2)

# find intersection
def find_intersect_r(pred_r, pred_norm, bins):
    if pred_r[0]>pred_norm[0]:
        return None
    else:
        for i in range(len(pred_r)):
            if pred_r[i]>=pred_norm[i]:
                return bins[i]

def find_intersect_l(pred_l, pred_norm, bins):
    pred_l = pred_l[::-1]
    pred_norm = pred_norm[::-1]
    bins = bins[::-1]
    if pred_l[0]>pred_norm[0]:
        return None
    else:
        for i in range(len(pred_l)):
            if pred_l[i]>=pred_norm[i]:
                return bins[i]


#%% Ensemble DBSCAN clustering
def ensemble_DBSCAN(X, esp_range=np.arange(0.1,0.6,0.1), min_smpl_range=np.arange(20,35,5)):
    """
    Clustering using multiple DBSCAN with different parameters.
    Majority vote at the end.
    """
    # record label and center
    cls_lib = dict()
    # create multiple DBSCAN
    for esp in esp_range:
        for min_smpl in min_smpl_range:
            db = DBSCAN(esp=esp,min_samples=min_smpl)


#%% Density gradient based 
def dense_clustering(X, nb_KNN=100, noise_level=0.05, eps = 0.5, min_smpl=60, mark_outlier=True, resolution=0.005, den_func=None):
    """
    (1) use KMean to find out north east first
    (2) calculate KNN distance
        space: O(n**2)
    (3) add density as feature (2D to 3D space)
    (4) mask data with noise level using density threshold
    (5) run DBSCAN on masked data
    (6) find top 3 size clusters
    (7) for each cluster
    (8) project onto PC space (This step can be skip if we are using max prob thres instead of 3 std thres)
    (9) fit a multivariate normal
    (10) for each data points in mask but in DBSCAN outlier, calculate the probability of these 4 multi-normal
    (11) give it to the highest prob
    (12) for northeast, since it used KMean directly, use std thresholding to select once again
    (13) label the rest as outlier
    """
    # parameter setting
    # nb_KNN = 30 # K nearest neighbor
    sigma = 1 # scaling for calculating density, the smaller the more emphasize on having close neighbor
    # noise_level = 0.05 # assumption of the percentage of noise inside dataset
    # resolution = 0.05 # stepsize for decreasing noise_level when mark_outlier=True
    # eps = 0.5 # DBSCAN parameter
    # min_smpl = 30 # DBSCAN parameter
    thres_attempt = 10 # threshold for DBSCAN attempt
    thres_nbCls = 100 # number of samples to be consider as a cluster
    multi_norm_lib = [] # record multi norm for each cluster
    pca_lib = [] # record pca model for each cluster
    multi_norm_lib_raw = [] # record multi norm for each cluster in raw space
    centroid_lib = [] # record centroid
    if den_func is None:
        # den_func =lambda x: np.median(np.exp(-x/sigma),axis=1) # to emphasize the close one
        den_func = lambda x: np.log(1/np.median(x,axis=1))

    # record group
    group_label = np.ones((X.shape[0],))*-1
    group_count = 0

    # create KD tress
    kdt = KDTree(X,metric='euclidean')
    # find k-distance
    dist, ind = kdt.query(X, k=nb_KNN)
    # calculate density
    den = den_func(dist)
    # use density as feature and feat DBSCAN
    feat = np.hstack([X,den.reshape(-1,1)])

    # KMean to get northest group
    fit_kmean = cluster.KMeans(n_clusters=4) # use 4 seeds as prior knowledge
    label_kmean = fit_kmean.fit_predict(feat)
    center_kmean = fit_kmean.cluster_centers_

    # find northest group
    cls_label = label_kmean==np.argmax([np.linalg.norm(x) for x in center_kmean])
    cls_data = X[cls_label,:]
    # record centroid
    centroid_lib.append([np.mean(cls_data[:,0]),np.mean(cls_data[:,1])])
    # fit multi norm in raw space    
    cls_mean = np.mean(cls_data,axis=0)
    cls_cov = np.cov(cls_data,rowvar=0)
    multi_norm = multivariate_normal(cls_mean,cls_cov)
    multi_norm_lib_raw.append(multi_norm)
    # project on to PC space
    cls_pca = PCA(n_components=2)
    cls_pca_data = cls_pca.fit_transform(cls_data)
    pca_lib.append(cls_pca)
    # fit multivariate normal
    cls_mean = np.mean(cls_pca_data,axis=0)
    cls_cov = np.cov(cls_pca_data,rowvar=0)
    multi_norm = multivariate_normal(cls_mean,cls_cov)
    multi_norm_lib.append(multi_norm)
    group_label[cls_label] = group_count
    group_count+=1

    # TODO: truncate northeast group in case northeast group has a high density
    # mask out noisy data (quantile on ~group 0 only)
    mask = den>np.quantile(den[~cls_label],noise_level)
    mask_data = X[mask,:]
    mask_feat = feat[mask,:]
    # repeat measurement until having more than 3 clusters
    centroid = []
    nb_smpl = []
    cls_i = []
    count_attempt = 0
    while len(centroid) <3 and count_attempt < thres_attempt:
        fit_model = cluster.DBSCAN(eps=eps, min_samples=min_smpl)
        label_model = fit_model.fit_predict(mask_feat)
        # record centroid and number of sample
        for i in set(label_model):
            if i!= -1:
                # set threshold for sample in cluster
                if sum(label_model==i) > thres_nbCls:
                    centroid.append([np.mean(mask_data[label_model==i,0]),np.mean(mask_data[label_model==i,1])])
                    nb_smpl.append(sum(label_model==i))
                    cls_i.append(i)
        if len(centroid) < 3:
            centroid = []
            nb_smpl = []
            cls_i = []
            # redirect DBSCAN
            if sum(nb_smpl) < X.shape[0]/2:
                print("enter redirect nb_smpl<X/2")
                # eps+=0.05
                min_smpl-=5
            else:
                print("enter redirect nb_smpl>X/2")
                # eps-=0.05
                min_smpl+=5
    # reset attempt count
    count_attempt = 0
    # report error if no enough cluster found
    if len(centroid) < 3:
        print("DBSCAN Error: cannot find enough clusters.")
        return None
    # select top 3 clusters 
    sort_cls = sorted(zip(centroid,nb_smpl,cls_i),key=lambda x: x[1], reverse=True)
    # for each top 3 clusters, pca -> fit multi_normal
    for tmp_cls in sort_cls[:3]:
        cls_label = tmp_cls[-1]
        # extract cluster data
        cls_data = mask_data[label_model==cls_label]
        # record centroid
        centroid_lib.append([np.mean(cls_data[:,0]),np.mean(cls_data[:,1])])
        # fit multi_normal in raw space
        cls_mean = np.mean(cls_data,axis=0)
        cls_cov = np.cov(cls_data,rowvar=0)
        multi_norm = multivariate_normal(cls_mean,cls_cov)
        multi_norm_lib_raw.append(multi_norm)
        # project on to PC space
        cls_pca = PCA(n_components=2)
        cls_pca_data = cls_pca.fit_transform(cls_data)
        pca_lib.append(cls_pca)
        # fit multivariate normal
        cls_mean = np.mean(cls_pca_data,axis=0)
        cls_cov = np.cov(cls_pca_data,rowvar=0)
        multi_norm = multivariate_normal(cls_mean,cls_cov)
        multi_norm_lib.append(multi_norm)
        # record group_label
        trunc_idx = np.nonzero(mask)[0][label_model==cls_label] # back to raw data level
        group_label[trunc_idx] = group_count
        group_count+=1

    # calculate the probability for all the outlier and cluster then into the highest prob
    # the rest of clusters
    # ===============================================================================
    # Prob based
    # group_label = prob_vote(X,group_label,mask,noise_level,multi_norm_lib_raw)
    # ===============================================================================
    # Distance based
    # group_label = dist_vote(X,group_label,mask,centroid_lib)
    # ===============================================================================
    # majority vote based
    outlier_count = sum(group_label==-1)
    while any(group_label==-1):
        group_label = maj_vote(group_label,dist,ind,mask_only=True,mask=mask)
        if outlier_count == sum(group_label==-1):
            break
        else:
            outlier_count = sum(group_label==-1)
    # ===============================================================================
    #majority vote for the outlier
    if mark_outlier:
        # step by step lowering noise_level
        noise_level -= resolution
        while noise_level > 0:
            mask = den>np.quantile(den[~cls_label],noise_level)
            while any(group_label==-1):
                group_label = maj_vote(group_label,dist,ind,mask_only=True, mask=mask)
                if outlier_count == sum(group_label==-1):
                    break
                else:
                    outlier_count = sum(group_label==-1)
            noise_level -= resolution
        # finished up the rest
        while any(group_label==-1):
            group_label = maj_vote(group_label,dist,ind,mask_only=False)
            if outlier_count == sum(group_label==-1):
                break
            else:
                outlier_count = sum(group_label==-1)

    return group_label, np.array(centroid_lib)

def maj_vote(group_label,dist,ind,nb_neighbor=100,mask_only=False,mask=None):
    tmp_label = []
    for d_i in range(len(group_label)):
        if mask_only and not mask[d_i]:
            continue
        if group_label[d_i]==-1:
            tmp_neighbor = ind[d_i][:nb_neighbor]
            if len(tmp_neighbor) > 0:
                # record noise distance
                noise_dist = np.mean(dist[d_i][group_label[tmp_neighbor]==-1])
                # ignore noise
                neighbor_cls = [group_label[x] for x in tmp_neighbor if group_label[x]!=-1]
                neighbor_dist = [dist[d_i][i] for i,x in enumerate(tmp_neighbor) if group_label[x]!=-1]

                if len(neighbor_cls)==0:
                    tmp_label.append(-1)
                else:
                    # majority vote
                    if True:
                        tmp_cls, tmp_vote = np.unique(neighbor_cls,return_counts=True)
                        tmp_label.append(tmp_cls[np.argmax(tmp_vote)])
                    else:
                        # TODO: weighted vote
                        tmp_count = []
                        tmp_vote= []
                        for c_i in range(4):
                            tmp_idx = np.array(neighbor_cls)==c_i
                            tmp_count.append(sum(tmp_idx))
                            tmp_dist = np.array(neighbor_dist)[tmp_idx]
                            if len(tmp_dist)!=0:
                                tmp_dist = np.mean(tmp_dist)
                                tmp_vote.append(1/tmp_dist)
                            else:
                                tmp_vote.append(-1)
                        if sum(tmp_vote)>-4:
                            tmp_label.append(np.argmax(tmp_vote))
                        else:
                            tmp_label.append(-1)
            else:
                tmp_label.append(-1)
    # assign group by majority vote
    if mask_only:
        group_label[(group_label==-1)*(mask)] = tmp_label    
    else:
        group_label[group_label==-1] = tmp_label

    return group_label

def dist_vote(X,group_label,mask,centroid_lib):
    for d_i in range(len(group_label)):
        if group_label[d_i]==-1 and mask[d_i]:
        # if group_label[d_i]==-1:
            tmp_X = X[d_i].reshape(1,-1)
            tmp_dist = np.Inf
            for i,tmp_center in enumerate(centroid_lib):
                loc_dist = np.linalg.norm(tmp_X-tmp_center)
                if loc_dist < tmp_dist:
                    tmp_dist = loc_dist
                    group_label[d_i] = i
    return group_label

def prob_vote(X,group_label,mask,noise_level,multi_norm_lib_raw):
    for d_i in range(X.shape[0]):
        if group_label[d_i]==-1 and mask[d_i]:
        # if group_label[d_i]==-1:
            tmp_X = X[d_i].reshape(1,-1)
            # calculate northest first since it has the largest variance and will take over all the outlier
            if multi_norm_lib_raw[0].pdf(tmp_X) >= noise_level:
                group_label[d_i] = 0
            tmp_prob = -1
            for i, tmp_multi in enumerate(multi_norm_lib_raw[1:]):
                # prob based
                loc_prob = tmp_multi.pdf(tmp_X)
                # if loc_prob > tmp_prob and loc_prob >= noise_level:
                if loc_prob > tmp_prob:
                    tmp_prob = loc_prob
                    group_label[d_i] = i+1
    return group_label








#%% Visualization
#%% UMAP
def plt_umap(pltdata, label_list=None):
    """
    visualize the data structure using UMAP
    Input:
      pltdata: data to plot. dimension: # of sample by features
      label_list: labels of each sample. Should have same lenth as # of sample
    Output:
      mapper: The UMAP embedding results
    """
    if len(pltdata.shape)==1:
        plt.scatter(label_list, pltdata)
        plt.grid(True)
        return None
    mapper = umap.UMAP().fit(pltdata)
    if label_list is not None:
        if type(label_list) is not np.ndarray:
            label_list = np.array(label_list)
        umap.plot.points(mapper, labels=label_list)
    else:
        umap.plot.points(mapper)
    return mapper

def plt_db(db, data, k=None):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    if k==None:
        for k, col in zip(unique_labels, colors):
            # k = 1
            # col = [1,1,1,1]
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            plt.plot(
                data[:, 0],
                data[:, 1],
                "o",
                markerfacecolor='k',
                markeredgecolor=None,
                markersize=0.1,
            )

            xy = data[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor=None,
                markersize=0.1,
            )

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor=None,
                markersize=1,
            )

            plt.title("Estimated number of clusters: %d" % n_clusters_)
            plt.show()
    else:
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        col = [1, 1, 1, 1]
        class_member_mask = labels == k

        plt.plot(
            data[:, 0],
            data[:, 1],
            "o",
            markerfacecolor='k',
            markeredgecolor=None,
            markersize=0.1,
        )

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=None,
            markersize=0.1,
        )

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=None,
            markersize=1,
        )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

#%% load dataset
def load_data(filepath,filename):
    raw_x=[]
    raw_y=[]
    with open(filepath+'/'+filename,'r') as f:
        for line in f:
            word = line.split('\t')
            raw_x.append(float(word[0]))
            raw_y.append(float(word[1].split()[0]))
    raw_data = np.hstack([np.array(raw_x).reshape((-1,1)),np.array(raw_y).reshape((-1,1))])
    return raw_data