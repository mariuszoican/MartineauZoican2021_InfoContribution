import sys
import time
import warnings
import pandas as pd
from gensim import corpora, models

warnings.filterwarnings("ignore", category=UserWarning)

import train_models.auxiliary_functions as aux


if __name__ == "__main__":
    start_time =  time.time()

    # load meta-data file
    df = pd.read_csv("nlp_metadata.csv")

    df['industry'] = df['ggroup']
    df['quarter_year'] = df["year"].astype("str") + " q" + df["quarter"].astype("str")

    df2 = df.dropna(subset=['industry']).reset_index(drop=True)
    df2['industry-quarter'] = list(zip(df2.industry, df2.quarter_year))
    list_industries_quarter = df2.groupby('industry-quarter').count().reset_index()['industry-quarter'].tolist()


    # generate the LDA Model
    print("Load LDA model...")
    path='./pretrained_models/'
    num_topics=60
    lda_model = models.LdaModel.load(path+"lda_%i.model"%int(num_topics))

    iq=list_industries_quarter[int(sys.argv[1])]
    print(iq)
    loop_time = time.time()

    industry=int(iq[0])
    quarter=iq[1]

    data_industry_quarter = aux.get_shapley(df, industry, quarter,lda_model,num_topics)
    data_industry_quarter = data_industry_quarter.merge(
        df[(df['industry'] == industry) & (df['quarter_year'] == quarter)][
        ['Analyst', 'GenderAnalyst', 'Contributor']].drop_duplicates(), on='Analyst')

    print("--- %s seconds ---" % (time.time() - loop_time))

    data_industry_quarter.to_csv('.\OutputShapley\Data_ShapleyR_%i_%s_%s.csv'%(int(sys.argv[1]),str(industry),str(quarter)))
