
import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
def pca(dataset, top_n_feature):
    #均值向量
    mean_ = np.mean(dataset, axis=0)
    #标准化
    dataset = dataset - mean_  # remove mean
    #协方差矩阵
    cov_matrix = np.cov(dataset, rowvar=0)
    #特征值，特征矩阵
    eig_values, eig_vectors = np.linalg.eig(np.mat(cov_matrix))
    #按特征值从大到小排序
    eig_index = np.argsort(-eig_values)
    eig_index=eig_index[:top_n_feature]
    principal = eig_vectors[:, eig_index]
    result = dataset * principal
    return result
# import some data to play with
iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = pca(iris.data,3)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
# PCA结束
# ------------------------------------------------------------------------------------
# SVD开始
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import glob


def loadIMageSet(data):
    filenames = glob.glob(data)
    filenames.sort()
    img = [Image.open(fn).convert('L').resize((100, 100)) for fn in filenames]
    face_image = np.asarray([np.array(im) for im in img])
    face_data = np.asarray([np.array(im).flatten() for im in img])
    return face_data, face_image


def SVD(data_, k):
    '''
    :param data_: 数据路径
    :param k: 主成分个数
    :return: 降维后的特征脸数据,原图数据,均值脸
    '''
    face_data, face_img = loadIMageSet(data_)
    # 均值脸
    face_avg = mean(face_data, 0)
    nSample, ndim = face_data.shape
    # svd
    u, s, v = np.linalg.svd(face_data)
    # 数据重构

    U = np.zeros((nSample, k))
    V = np.zeros((k, ndim))
    S = np.zeros((k, k))
    for i in range(k):
        U[:, i] = u[:, i]
        V[i, :] = v[i, :]
        S[i, i] = s[i]
    data = np.dot(np.dot(U, S), V)
    return data, face_img, face_avg


def draw():
    fig, axes = plt.subplots(3, 5, figsize=(4, 5), subplot_kw={"xticks": [], "yticks": []})
    for i, ax in enumerate(axes.flat):
        V, face_img, face_avg = SVD('../Lab03/FaceData/*.pgm', i+1)
        ax.imshow(V[14, :].reshape(100, 100))
    plt.show()
    return face_avg


if __name__ == '__main__':
    # SVD
    face_avg = draw()
    # 均值脸
    plt.imshow(face_avg.reshape(100, 100), cmap="gray")
    plt.show()
