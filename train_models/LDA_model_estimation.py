import pandas as pd
from gensim import models
import time
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# load auxiliary functions
import auxiliary_functions as aux
    


if __name__ == "__main__":

    notopics = 60

    print("Number of topics: ", notopics)

    start_time =  time.time()

    print("Load metadata...")
    df = pd.read_csv("../nlp_metadata.csv")
    df['industry'] = df['ggroup']
    df['quarter_year'] = df["year"].astype("str") + " q" + df["quarter"].astype("str")
    all_companies=df.groupby('TICKER')["DCN"].count().reset_index()['TICKER'].tolist()
    dcns = set(df["DCN"])
    all_files_dcns = aux.get_company_files_training(dcns)
    analyst_to_index = aux.construct_analyst_index_mapping(df, all_files_dcns)

    meta_time =  time.time()

    words=[]
    did=0

    print("Metadata loaded:",meta_time-start_time)
    
    print("Tokenize and build corpus....")
    words, dictionary_LDA, corpus=aux.construct_corpus(all_files_dcns)


    perplexity_values = []
    model_list = []

    df_coherence = pd.DataFrame(columns=['No topics', 'Log perplexity'])

    loop_time=time.time()
    print("----- Number of topics: ",notopics)
    lda_model = models.LdaModel(corpus=corpus,
                                    id2word=dictionary_LDA,
                                    num_topics=notopics,
                                    random_state=100,
                                    chunksize=2000,
                                    passes=20,
                                    alpha=0.1,
                                    eta=0.01)
    lda_model.save('../pretrained_models/lda_%i.model'%int(notopics))
    loop_end=time.time()
    print("Loop time: ", loop_end-loop_time)


    perp=lda_model.log_perplexity(corpus)
    perplexity_values.append(perp)
    print("-------- Get perplexity: ", perp)

    new_row = {'No topics':notopics,
               'Log perplexity': perp}

    df_coherence = df_coherence.append(new_row, ignore_index=True)
    df_coherence.to_csv('Output/perplexity_fullsample_%i.csv'%int(notopics))