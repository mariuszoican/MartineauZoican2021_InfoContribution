import pandas as pd
import numpy as np
from gensim import models
import time
import train_models.auxiliary_functions as aux

DataShapley = pd.read_csv('DataShapley.csv', index_col=0)
DataShapley['TickerQuarter']=list(zip(DataShapley.Ticker, DataShapley.Quarter))
list_tickerquarter=DataShapley['TickerQuarter'].drop_duplicates().to_list()

df = pd.read_csv("nlp_metadata.csv")
df['Ticker'] = df['TICKER']

df['industry'] = df['ggroup']
df['quarter_year'] = df["year"].astype("str") + " q" + df["quarter"].astype("str")

print("Load LDA model...")
path = './pretrained_models/'
num_topics = 60
lda_model = models.LdaModel.load(path + "lda_%i.model" % int(num_topics))

list_topicnames = ['Ticker', 'Quarter', 'Analyst', 'GenderAnalyst', 'Contributor'] + [
    'Topic %i' % x for x in range(num_topics)]

def get_data_topics(ticker,quarter):
    data_topics = pd.DataFrame(columns=list_topicnames)

    temp_df=df[(df.TICKER==ticker) &
               (df.quarter_year==quarter)].drop_duplicates()[['Analyst','Contributor','GenderAnalyst','DCN']]
    list_dcns = temp_df['DCN'].to_list()
    list_analyst=temp_df['Analyst'].to_list()
    list_gender=temp_df['GenderAnalyst'].to_list()
    list_contr=temp_df['Contributor'].to_list()

    for l in list_dcns:
        #print(l)
        words, dictionary_LDA, corpus=aux.construct_corpus(aux.get_company_files([l]))
        matrix = []
        row = [0] * num_topics
        if len(words)==0:
            continue
        temp_bow = lda_model.id2word.doc2bow(words[0])
        topics = lda_model[temp_bow]
        for index, dist in topics:
            row[index] = dist
        matrix.append(row)
        matrix = np.array(matrix).mean(axis=0)

        anl=list_analyst[list_dcns.index(l)]
        gndr=list_gender[list_dcns.index(l)]
        cont=list_contr[list_dcns.index(l)]

        new_row = [ticker, quarter,anl,gndr,cont] + list(matrix)
        data_topics.loc[len(data_topics)] = new_row

    data_topics=data_topics.dropna(subset=['Analyst'])

    return [data_topics.groupby(['Ticker','Quarter','Analyst']).mean().reset_index()
        ,data_topics['Analyst'].to_list()]

def topics_analyst(anl,data_topics):
    if anl==np.nan:
        return -1
    return [data_topics['Ticker'][0], data_topics['Quarter'][0], anl,
            data_topics[data_topics['Analyst']==anl][['Topic %s'%str(x) for x in [21,30]]].sum(axis=1).mean()]

data_technical=pd.DataFrame(columns=['Ticker','Quarter','Analyst','WeightTechnical'])
for (ticker, quarter) in list_tickerquarter:
    print(ticker,quarter)
    data_topics, list_anl = get_data_topics(ticker,quarter)

    for anl in list_anl:
        row=topics_analyst(anl,data_topics)
        data_technical.loc[len(data_technical)]=row

data_technical.to_csv('DataShapley_TechnicalTopicWeights.csv')