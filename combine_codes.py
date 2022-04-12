import pandas as pd
from sklearn import preprocessing

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#merge 하고 싶은 file list들
file_list = ["/home/ylab2/Downloads/body/data/mc365.xlsx","/home/ylab2/Downloads/body/data/sizekorea.xlsx","/home/ylab2/Downloads/body/data/3d_160.xlsx"]
merge_df = pd.DataFrame()
file_df2 = pd.DataFrame()

for file_name in file_list:
    file_df = pd.read_excel(file_name,engine='openpyxl')
    file_df2 = file_df2.append(file_df, ignore_index=True)

#모든 column 활용할 경우
columns = list(file_df.columns)

temp_df = pd.DataFrame(file_df2, columns = columns)
merge_df = merge_df.append(temp_df, ignore_index=False)
# merge_df.to_csv("/home/ylab2/Downloads/body/data/merge_3d.csv", index=False, encoding ='utf-8-sig')
# df_3d = pd.read_csv("C:/Users/user/Desktop/excel/123/data/merge_3d.csv",encoding ="UTF-8")

#전처리 시작
merge_df.set_index('고객번호', inplace=True)
merge_df = merge_df.drop('스캔일자',axis=1)
merge_df = merge_df.dropna(how='all')  # if in row has null drop
merge_df = merge_df.dropna(axis=1)  # if in column have null drop
merge_df = merge_df.drop('출생년도',axis=1)  # '출생년도' is not exact info

#merge_df['출생년도'] = 2022 - merge_df['출생년도']
merge_df['성별'].replace({'남':0,'여':1}, inplace=True)

#남녀 데이터셋 만들기
merge_df['성별'].value_counts()

groups = merge_df.groupby(merge_df.성별)
female_df = groups.get_group(1)
male_df = groups.get_group(0)

# df.set_index('고객번호', inplace=True)

female_df.to_csv("/home/ylab2/Downloads/body/data/female.csv", index=True, encoding ='utf-8-sig')
male_df.to_csv("/home/ylab2/Downloads/body/data/male.csv", index=True, encoding ='utf-8-sig')



#famale(1007)
female = pd.read_csv("/data/female.csv")
female.set_index('고객번호', inplace=True)
female = female.drop('성별', axis=1)

for i in range(5,len(female)+1):
    female.iloc[:,i] = female.iloc[:,i]/female['신장']

female = female.drop('신장',axis=1)

a_row = list()
a_row = female.index

# PCA 주성분분석 (find n_components )
pca = PCA(random_state=1107)
X_p = pca.fit_transform(female)
a=pd.Series(np.cumsum(pca.explained_variance_ratio_))

# 주성분분석
pca = PCA(n_components=5)
printcipalComponents = pca.fit_transform(female)
female_pca = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2','component3', 'component4', 'component5'])
female_pca = female_pca.set_index(a_row)

female_pca.head()
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
female_pca.to_csv("/home/ylab2/Downloads/body/data/female_pca_t.csv")


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

female_pca.to_csv("/home/ylab2/Downloads/body/data/female_cluster.csv", encoding ='utf-8-sig')
