import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#famale
female_pca = pd.read_csv("/data/female/female_pca_t.csv", encoding ='utf-8-sig')
female_pca.set_index('고객번호', inplace=True)


#find cluster k by using elbow method
ks = range(1, 10)
inertias = []

print(female_pca.shape)

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(female_pca)
    inertias.append(model.inertia_)

cluster_count = 1
for i in inertias:
    print("Number of Cluster: %d" % cluster_count)
    print("inertia: " + str(i))
    cluster_count += 1

plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel('Number of Cluster')
plt.ylabel('Distortion')
plt.show()

# analysis cluster
k = 4

# 그룹 수, random_state 설정
model = KMeans(n_clusters = k, random_state = 10)

# 정규화된 데이터에 학습
model.fit(female_pca)

# 클러스터링 결과 각 데이터가 몇 번째 그룹에 속하는지 저장
female_pca['cluster'] = model.fit_predict(female_pca)

female_pca.to_csv("/data/female/female_cluster.csv", encoding ='utf-8-sig')



