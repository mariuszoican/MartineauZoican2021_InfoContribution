import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models, similarities

dict_industries={
    1010: 'Energy',
    1510: 'Materials',
    2010: 'Capital Goods',
    2020: 'Commercial and Professional Services',
    2030: 'Transportation',
    2510: 'Automobiles and Components',
    2520: 'Consumer Durables and Apparel',
    2530: 'Consumer Services',
    2550: 'Retailing',
    3010: 'Food and Staples Retailing',
    3020: 'Food, Beverage, and Tobacco',
    3030: 'Household and Personal Products',
    3510: 'Health Care Equipment and Services',
    3520: 'Pharmaceuticals, Biotech, and Life Sciences',
    4010: 'Banks',
    4020: 'Diversified Financials',
    4030: 'Insurance',
    4510: 'Software and Services',
    4520: 'Technology, Hardware and Equipment',
    4530: 'Semiconductors',
    5010: 'Communication Services',
    5020: 'Media and Entertainment',
    5510: 'Utilities',
    6010: 'Real Estate'
}


IndustryTopics=pd.read_csv('topic_loadings_by_industryquarter.csv',index_col=0)
IndustryMeans=IndustryTopics.groupby('Industry').mean()

def industry_topics(industry_code):
    SortedMeans=IndustryMeans.loc[industry_code].sort_values(ascending=False)
    topics=[x.replace('Topic ', '') for x in SortedMeans.reset_index()['index'].to_list()]
    loadings=SortedMeans.reset_index()[industry_code]
    return topics,loadings

def get_loadings(x):
    a, b = x.split("_")
    a=int(a)
    SortedMeans = IndustryMeans.loc[a].sort_values(ascending=False)
    return SortedMeans.loc[b]


for ind in list(dict_industries.keys()):
    plt.clf()
    plt.figure(figsize=(16, 8))
    plt.bar(industry_topics(ind)[0], industry_topics(ind)[1])
    plt.title(dict_industries[ind], fontsize=18)
    plt.savefig('./graphs_industry/topics_%s.pdf'%dict_industries[ind],bbox_inches='tight')

# generate the LDA Model
print("Load LDA model...")
path = '../pretrained_models/'
num_topics = 60
lda_model = models.LdaModel.load(path + "lda_%i.model" % int(num_topics))


IndTransp=IndustryMeans.transpose()
top_loadings=pd.DataFrame(data=dict_industries.keys())
top_loadings=top_loadings.rename(columns={0:'Industry'})
top_loadings['First Topic']=top_loadings['Industry'].apply(lambda x: IndTransp[x].nlargest(2).reset_index()['index'][0])
top_loadings['Second Topic']=top_loadings['Industry'].apply(lambda x: IndTransp[x].nlargest(2).reset_index()['index'][1])
top_loadings=top_loadings.set_index('Industry').stack().reset_index()
top_loadings=top_loadings.rename(columns={'level_1': 'Top Topic', 0: 'Topic'})
top_loadings['Industry name']=top_loadings['Industry'].apply(lambda x: dict_industries[x])
top_loadings['words']=top_loadings['Topic'].apply(lambda x: lda_model.show_topic(int(x.split(" ")[1])))

top_loadings['Industry_Topic']=top_loadings['Industry'].map(str)+"_"+top_loadings['Topic']

top_loadings['Loading']=top_loadings['Industry_Topic'].apply(lambda x: get_loadings(x))
del top_loadings['Industry_Topic']
top_loadings=top_loadings[['Industry', 'Industry name', 'Top Topic', 'Topic', 'words', 'Loading']]
top_loadings.to_csv('TopTopics_Industries.csv')