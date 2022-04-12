import pandas as pd
from scipy.spatial import distance

female = pd.read_csv('/data/female/female_data_t.csv')

female = female.drop('성별', axis=1)
female = female.dropna(how='all')# if in row has null drop
female.set_index('고객번호',inplace=True)

female.to_csv('/home/ylab2/Downloads/body/data/female/female_data_total.csv', index=True, encoding ='utf-8-sig')

male = pd.read_csv('/data/male/male_data_t.csv')
male = male.drop('성별', axis=1)
#um....compare each row of dataframe female..
# total 1007 row

def calSimilarity(a):
    # 비교하고 싶은 값을 a 라고 하자
    min_value = 100
    k = 0

    for i in range(0,1007):
        dist = distance.euclidean(male.iloc[a], male.iloc[i])
        if dist == 0:
            continue

        if min_value > dist:
            min_value = dist
            k = i
            print(k)

    return k, min_value

m_1 = calSimilarity(1)
m_504 = calSimilarity(504)
m_490 = calSimilarity(490)
m_10 = calSimilarity(10)
m_429 = calSimilarity(429)
