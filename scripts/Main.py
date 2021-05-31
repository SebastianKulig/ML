import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def silhouette_for_every_sample(feature_list, labels_list, number_of_clusters):
    silhouette_avg = silhouette_score(feature_list, labels_list)

    # ================== Compute the silhouette scores for each sample ==============
    sample_silhouette_values = silhouette_samples(feature_list, labels_list)
    y_lower = 10
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    for i in range(number_of_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels_list == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / number_of_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

def show_plots(table, clustering_method_name):
    # ========================================= Silhouette - plot ======================================================
    plt.plot([i[1] for i in table], [i[2] for i in table], 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title(clustering_method_name)
    plt.show()


# ================================================= Settings ============================================================
model_name = "ResNet50"
dataset = ""
clustering_method = "k-means"

result_folder = "C:\\Users\\kulig\\Desktop\\ML-projekt\\results\\"

max_cluster_number = 10

silhouette_avg_table = []
calinski_harabasz_table = []
davies_bouldin_table = []

# ================================================= ResNet50 ============================================================
model_ResNet50 = ResNet50(weights='imagenet', include_top=False)
model_ResNet50.summary()
#
# # ====================================================VGG16==============================================================
model_VGG16 = VGG16(weights='imagenet', include_top=False)
model_VGG16.summary()

# ================================================= Choose Modele =======================================================
if (model_name == "ResNet50"):
    model = model_ResNet50
elif (model_name == "VGG16"):
    model = model_VGG16

# ================================================== Choose Dataset =====================================================
if (dataset == "mini"):
    img_path = '../dataset/mini_data\\'
else:
    img_path = '../dataset/natural_images\\'

# ================================================= Extract features ===================================================
vgg16_feature_list = []
# for fname in os.listdir(img_path):
#     img_path_full = img_path + fname
#     img = image.load_img(img_path_full, target_size=(224, 224))
#     img_data = image.img_to_array(img)
#     img_data = np.expand_dims(img_data, axis=0)
#     img_data = preprocess_input(img_data)
#
#     vgg16_feature = model.predict(img_data)
#     vgg16_feature_np = np.array(vgg16_feature)
#     vgg16_feature_list.append(vgg16_feature_np.flatten())


vgg16_feature_list_np = np.array(vgg16_feature_list)

# =============================== temporary ==================================
# with open("../cechy.npy", "wb") as f:
#     np.save(f, vgg16_feature_list_np)

with open("C:\\Users\\kulig\\Desktop\\ML-projekt\\cechy.npy", "rb") as f:
    vgg16_feature_list_np = np.load(f)
# ===================================================K-means=============================================================

inertia = []
K = range(2, max_cluster_number)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(vgg16_feature_list_np)
    inertia.append(kmeanModel.inertia_)
    y_kmeans = kmeanModel.fit_predict(vgg16_feature_list_np)

    # ================================ Check the performance of clustering =================================================
    # ==================================== Silhouette ===============================
    silhouette_avg = silhouette_score(vgg16_feature_list_np, y_kmeans)
    silhouette_avg_table.append([model_name, k, silhouette_avg])

    # ================================ Calinski-Harabasz ============================
    calinski_harabasz = metrics.calinski_harabasz_score(vgg16_feature_list_np, y_kmeans)
    calinski_harabasz_table.append([model_name, k, calinski_harabasz])

    # ================================ Davies-Bouldin ===============================
    davies_bouldin = metrics.davies_bouldin_score(vgg16_feature_list_np, y_kmeans)
    davies_bouldin_table.append([model_name, k, davies_bouldin])


    #silhouette_for_every_sample(vgg16_feature_list_np, y_kmeans, k)


show_plots(silhouette_avg_table, "Silhouette")
show_plots(calinski_harabasz_table, "Calinski-Harabasz")
show_plots(davies_bouldin_table, "Davies-Bouldin")


    # =============================================== Plot the elbow ========================================================
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Wykres łokciowy')
plt.show()

print(silhouette_avg_table)
print(calinski_harabasz_table)
print(davies_bouldin_table)


# =========================================== Save results in separate folders =========================================

# result_folder = "C:\\Users\\kulig\\Desktop\\ML-projekt\\results\\"
# onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]

# for i in range(optimal_number_of_clusters):
#     folder_name = result_folder + str(i)
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#         print("Directory ", folder_name, " Created ")
#     else:
#         print("Directory ", folder_name, " already exists")

# i = 0
# for file in onlyfiles:
#     src_dir = img_path + file
#     dst_dir = result_folder + str(y_kmeans[i])
#     shutil.copy(src_dir, dst_dir)
#     i += 1
