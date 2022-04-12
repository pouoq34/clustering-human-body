import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#famale(1007)
female = pd.read_csv("//data/female_data_t.csv")
female.set_index('고객번호', inplace=True)
a_row = list()
a_row = female.index

# PCA 주성분분석 (find n_components )
pca = PCA(random_state=1107)
X_p = pca.fit_transform(female)
a=pd.Series(np.cumsum(pca.explained_variance_ratio_))

# 주성분분석
pca = PCA(n_components=14)
printcipalComponents = pca.fit_transform(female)
female_pca = pd.DataFrame(data=printcipalComponents, columns = ['component1', 'component2','component3', 'component4', 'component5', 'component6','component7',
                                                                 'component8','component9','component10','component11','component12','component13','component14'])
female_pca = female_pca.set_index(a_row)

female_pca.head()
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
female_pca.to_csv("/home/ylab2/Downloads/body/data/female_pca_t.csv")


#male(428)
male = pd.read_csv("//data/male_data_t.csv")
male.set_index('고객번호', inplace=True)

m_row = list()
m_row = male.index

# PCA 주성분분석 (find n_components )
m_pca = PCA(random_state=1107)
m_p = m_pca.fit_transform(male)
m = pd.Series(np.cumsum(m_pca.explained_variance_ratio_))

# 주성분분석
pca = PCA(n_components=17)
maleComponents = pca.fit_transform(male)
male_pca = pd.DataFrame(data=maleComponents, columns = ['component1', 'component2','component3', 'component4', 'component5', 'component6','component7',
                                                                 'component8','component9','component10','component11','component12','component13','component14','component15','component16','component17'])
male_pca = male_pca.set_index(m_row)

male_pca.to_csv("/home/ylab2/Downloads/body/data/male_pca_t.csv")