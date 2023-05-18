#%% import library
import dependencies as dep
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

#%% load data
filepath = "C:\\Users\\Yuan\\OneDrive\\Desktop\\Algorithm questionnaire\\2 algorithm Data set\\Q2"
filename = 'Dataset 5.txt'
savename = 'Q2 Answer 1.txt'
raw_x = []
raw_y = []
with open(filepath+'/'+filename,'r') as f:
    for line in f:
        word = line.split('\t')
        raw_x.append(float(word[0]))
        raw_y.append(float(word[1].split()[0]))
raw_data = np.hstack([np.array(raw_x).reshape((-1,1)),np.array(raw_y).reshape((-1,1))])

test_lib = []
for i in range(1,7):
    test_lib.append(dep.load_data(filepath,f"Dataset {i}.txt"))


#%% scatter plot
plt.figure()
plt.plot(raw_data[:,0], raw_data[:,1], '.', markersize=0.2)
idx = np.array([x&y&i for x,y,i in zip(raw_data[:,0]<4,raw_data[:,1]<3,raw_data[:,1]>-2)])
group1 = raw_data[idx,:]
plt.plot(group1[:,0], group1[:,1], '.', markersize=0.2)

#%% Ensemble clustering
X = test_lib[5]
label_model = dep.dense_clustering(X,noise_level=0.1,eps=0.5,min_smpl=60,mark_outlier=True)
plt.scatter(X[:,0], X[:,1],s=1,c='k')
for i in set(label_model):
    i = int(i)
    plt.plot(X[label_model==i,0],X[label_model==i,1],'.',markersize=1)
    plt.plot(np.mean(X[label_model==i,0]),np.mean(X[label_model==i,1]),'.',markersize=15)


#%% DBSCAN
"""
look most promising. Ensemble with different parameters > majority vote
eps = 2, min_samples=30, can capture northeast group
"""
db = dep.DBSCAN(eps=2, min_samples=30).fit(raw_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print(f"Estimated number of noise points/ number of points: {n_noise_}/ {raw_data.shape[0]}")

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print(
#     "Adjusted Mutual Information: %0.3f"
#     % metrics.adjusted_mutual_info_score(labels_true, labels)
# )
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(raw_data, labels))

# %% visualization
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
k = 1
col = [1,1,1,1]
if k == -1:
    # Black used for noise.
    col = [0, 0, 0, 1]

class_member_mask = labels == k

plt.plot(
    raw_data[:, 0],
    raw_data[:, 1],
    "o",
    markerfacecolor='k',
    markeredgecolor=None,
    markersize=0.1,
)

xy = raw_data[class_member_mask & core_samples_mask]
plt.plot(
    xy[:, 0],
    xy[:, 1],
    "o",
    markerfacecolor=tuple(col),
    markeredgecolor=None,
    markersize=0.1,
)

xy = raw_data[class_member_mask & ~core_samples_mask]
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

#%% kmean
"""
k mean can capture northeast
"""
fit_kmean = dep.cluster.KMeans(n_clusters=4)
embed_kmean = fit_kmean.fit_predict(raw_data)
center = fit_kmean.cluster_centers_
plt.figure
for i, c in zip(set(embed_kmean),center):
    plt.plot(raw_data[embed_kmean==i,0],raw_data[embed_kmean==i,1],'.',markersize=1)
    plt.plot(c[0],c[1],'.',markersize=10)

#%% agglomerative clustering
"""
Done the job perfectly for Dataset 1.
"""
fit_agg = dep.cluster.AgglomerativeClustering(n_clusters=4)
embed_agg = fit_agg.fit_predict(raw_data)
plt.figure
for i in set(embed_agg):
    plt.plot(raw_data[embed_agg==i,0],raw_data[embed_agg==i,1],'.',markersize=1)



# %% spectral clustering
"""
Takes too long
"""
# fit_spec = dep.cluster.SpectralClustering(n_clusters=4)
# embed_spec = fit_spec.fit_predict(raw_data)
# center = fit_spec.cluster_centers_
# plt.figure
# for i, c in zip(set(embed_spec),center):
#     plt.plot(raw_data[embed_spec==i,0],raw_data[embed_spec==i,1],'.',markersize=1)
#     plt.plot(c[0],c[1],'.',markersize=10)

"""
Both UMAP and TSNE fail.
"""
# #%% umap
# fit_umap = dep.umap.UMAP(n_neighbors=5,n_components=1)
# embedding = fit_umap.fit_transform(raw_data)
# g1 = fit_umap.transform(group1)
#%% vis umap
# plt.figure
# plt.plot(embedding, '.', markersize=0.2)
# plt.plot(g1, '.', markersize=0.2)

# #%% TSNE
# fit_tsne = dep.TSNE()
# embed_tsne = fit_tsne.fit_transform(raw_data)
# g1_tsne = fit_tsne.transform(group1)

# #%% vis tsne
# plt.figure
# plt.plot(embed_tsne[:,0],embed_tsne[:,1], '.', markersize=0.2)

#%% PCA preprocessing
"""
Let's try PCA then clusters. There is a hint saying that the relative position is not changing.
"""
std_scaler = dep.StandardScaler()
norm_data = std_scaler.fit_transform(raw_data)
plt.figure()
plt.plot(norm_data[:,0], norm_data[:,1], '.', markersize=0.2)

# %% PCA
pca = dep.PCA(n_components=2)
kernal_pca = dep.KernelPCA(n_components=2,kernel='rbf')
X_pca = pca.fit_transform(norm_data)

#%%
fit_kmean = dep.cluster.KMeans(n_clusters=4)
embed_kmean = fit_kmean.fit_predict(X_pca)
center = fit_kmean.cluster_centers_
plt.figure
for i, c in zip(set(embed_kmean),center):
    plt.plot(X_pca[embed_kmean==i,0],X_pca[embed_kmean==i,1],'.',markersize=1)
    plt.plot(c[0],c[1],'.',markersize=10)

#%% remove northeast
preserve_idx = embed_kmean!=np.argmax([np.linalg.norm(x) for x in center])
trunc_data = X_pca[preserve_idx]
trunc_pca = std_scaler.fit_transform(trunc_data)
trunc_pca = pca.fit_transform(trunc_pca)
fit_kmean = dep.cluster.KMeans(n_clusters=3)
embed_trunc = fit_kmean.fit_predict(trunc_pca)
center_new = fit_kmean.cluster_centers_
plt.figure
for i, c in zip(set(embed_trunc),center_new):
    plt.plot(trunc_data[embed_trunc==i,0],trunc_data[embed_trunc==i,1],'.',markersize=1)
    plt.plot(c[0],c[1],'.',markersize=10)

#%%
fit_db = dep.DBSCAN(eps=0.2, min_samples=30).fit(trunc_pca)
labels = fit_db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters: %d" % n_clusters_)
#%%
dep.plt_db(fit_db,trunc_pca,k=0)

#%%
cluster1 = trunc_data[labels==0]
# pls = dep.PLSRegression(n_components=2)
# cls1_pca = pls.fit_transform(cluster1)
cls1_pca = dep.PCA(n_components=1).fit_transform(cluster1)
n, bins= np.histogram(cls1_pca, bins=100)

# fit curve
opt_norm1,_ = dep.optimize.curve_fit(dep.gaussian,bins[:-1],n)

# examine left and right
right_idx = bins[:-1]>(opt_norm1[1]+opt_norm1[2]*2)
left_idx = bins[:-1]<(opt_norm1[1]-opt_norm1[2]*2)
opt_norm_right,_ = dep.optimize.curve_fit(dep.gaussian,bins[:-1][right_idx],n[right_idx])
# opt_norm_left,_ = dep.optimize.curve_fit(dep.gaussian,bins[:-1][left_idx],n[left_idx])

# visualization
plt.figure
plt.plot(bins[:-1],n)
plt.plot(bins[:-1],dep.gaussian(bins[:-1],*opt_norm1))
plt.plot(bins[:-1][right_idx],dep.gaussian(bins[:-1][right_idx],*opt_norm_right),linewidth=3)
# plt.plot(bins[:-1][left_idx],dep.gaussian(bins[:-1][left_idx],*opt_norm_left),linewidth=3)
plt.vlines(opt_norm1[1]+opt_norm1[2]*2, 0, 300)
plt.vlines(-(opt_norm1[1]+opt_norm1[2]*2), 0, 300)
plt.vlines(opt_norm_right[1], 0, 200, linewidth = 3)
plt.vlines(opt_norm_right[1]+opt_norm_right[2]*2, 0, 200, linewidth = 3)
plt.vlines(opt_norm_right[1]-opt_norm_right[2]*2, 0, 200, linewidth = 3)
# plt.vlines(opt_norm_left[1], 0, 200, linewidth = 3)
# plt.vlines(opt_norm_left[1]+opt_norm_left[2]*1, 0, 200, linewidth = 3)
# plt.vlines(opt_norm_left[1]-opt_norm_left[2]*1, 0, 200, linewidth = 3)

#%%
pred_norm1 = dep.gaussian(bins[:-1][right_idx],*opt_norm1)
pred_r = dep.gaussian(bins[:-1][right_idx],*opt_norm_right)
prob_intersect = dep.find_intersect_r(pred_r,pred_norm1,bins[:-1][right_idx])
new_cls_idx = (cls1_pca >= prob_intersect)
tmp = np.nonzero(labels==0)[0][new_cls_idx.reshape(-1)]
labels[tmp] = 2

#%%
group3 = raw_data[~preserve_idx]
rest = raw_data[preserve_idx]
group2 = rest[labels==1]
tmp = rest[labels==0]
group0 = tmp[~new_cls_idx.reshape(-1)]
group1 = tmp[new_cls_idx.reshape(-1)]

plt.figure()
plt.plot(group0[:,0],group0[:,1],'.',label='group0',markersize=0.5)
plt.plot(group1[:,0],group1[:,1],'.',label='group1',markersize=0.5)
plt.plot(group2[:,0],group2[:,1],'.',label='group2',markersize=0.5)
plt.plot(group3[:,0],group3[:,1],'.',label='group3',markersize=0.5)

#%%
test_data = test_lib[3]
group_lib, group_label, n_cluster = dep.cluster_pipe(test_data,eps=0.2,min_smpl=30)
print(f"n_cluster = {n_cluster}")
fig, ax = plt.subplots(2,1,figsize=(8,10),sharex=True,sharey=True)
for i, x in enumerate(group_lib):
    ax[0].plot(x[:,0],x[:,1],'.',label=f"cluster {i}",markersize=1)
ax[0].legend()
ax[1].plot(test_data[:,0],test_data[:,1],'.',label=f"cluster {i}",markersize=1)

# #%% TEST OPTICS
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
# clust = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.05)
# # Run the fit
# label_optics = clust.fit_predict(test_data)

# #%%
# plt.figure
# for i in set(label_optics):
#     plt.plot(test_data[label_optics==i,0],test_data[label_optics==i,1],'.',markersize=1)


#%% spectral clustering
# # plot test data
# for i,x in enumerate(test_lib):
#     plt.figure
#     plt.plot(x[:,0],x[:,1],'k.',markersize=1)
#     plt.tick_params(
#         axis='both',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         left=False,
#         right=False,
#         labelleft=False,
#         labelbottom=False) # labels along the bottom edge are off
#     plt.savefig(f"plot{i}")
#     plt.show()
#     plt.clf()

# from PIL import Image
# from matplotlib import image
# from sklearn.feature_extraction import image as sk_img
# from skimage.transform import resize

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# img = plt.imread('plot0.png')
# img = resize(img,(100,100))
# img = img[:,:,0]
# mask = img.astype(bool)
# plt.imshow(img,cmap='gray')
# plt.imshow(mask,cmap='gray')
# graph = sk_img.img_to_graph(img)
# graph.data = np.exp(-graph.data/graph.data.std())
# label_spec = dep.cluster.spectral_clustering(graph, n_clusters=4)
# label_im = np.full(mask.shape, -1.0)
# label_im = label_spec.reshape((100,100))
# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# axs[0].imshow(img,cmap='gray')
# axs[1].imshow(label_im,cmap='gray')

#%% Test KNN
X = test_lib[1]
noise_level = 0.05
group_label = np.ones((X.shape[0]))*-1
bt = dep.KDTree(X,leaf_size=40,metric='euclidean')
# find k
dist, ind = bt.query(X, k=100)
sigma = 1
# den = np.median(np.exp(-dist/sigma),axis=1) # to emphasize the close one
den = np.log(1/np.median(dist,axis=1))
mask = den>np.quantile(den,noise_level)
plt.figure
plt.scatter(X[:,0], X[:,1],s=2,c='k')
plt.scatter(X[mask,0], X[mask,1],s=1,c=den[mask])

# use density as feature
mask_data = X[mask,:]
feat = np.hstack([mask_data,den[mask].reshape(-1,1)])
fit_model = dep.cluster.DBSCAN(eps=0.5, min_samples=50)
# fit_model = dep.cluster.KMeans(n_clusters=4)
label_model = fit_model.fit_predict(feat)
centroid = []
nb_smpl = []
cls_i = []
plt.figure()
plt.scatter(X[:,0], X[:,1],s=1,c='k')
for i in set(label_model):
    if i!= -1:
        plt.plot(mask_data[label_model==i,0],mask_data[label_model==i,1],'.',markersize=1)
        plt.plot(np.mean(mask_data[label_model==i,0]),np.mean(mask_data[label_model==i,1]),'.',markersize=20)
        centroid.append([np.mean(mask_data[label_model==i,0]),np.mean(mask_data[label_model==i,1])])
        nb_smpl.append(sum(label_model==i))
        cls_i.append(i)

#%% 
left_most = mask_data[label_model==0]
# project on to PC space
lm_pca = dep.PCA(n_components=2)
lm_pca_data = lm_pca.fit_transform(left_most)
# fit multivariate normal
lm_mean = np.mean(lm_pca_data,axis=0)
lm_cov = np.cov(lm_pca_data,rowvar=0)
multi_norm = dep.multivariate_normal(lm_mean,lm_cov)
# find prob at 3 std on x-axis and y-axis
x_test = lm_mean+lm_cov[0,:]*2
y_test = lm_mean+lm_cov[1,:]*2
prob_thres = np.mean([multi_norm.pdf(x_test),multi_norm.pdf(y_test)])
# prob_thres = 0.9
# all points with probability smaller than thres are in left most cluster
test_prob = multi_norm.pdf(lm_pca.transform(X))
group_label = np.ones(X.shape[0])*-1
group_label[test_prob>=prob_thres] = 0
plt.figure()
plt.scatter(X[:,0], X[:,1],s=1,c='k')
plt.scatter(X[group_label==0,0], X[group_label==0,1],s=1,c='b')


# %%
