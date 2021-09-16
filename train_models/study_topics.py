import pandas as pd
import numpy as np
from gensim import models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, font_manager
import seaborn as sns
import numpy.random as rnd
import datetime as dt
#from linearmodels.panel import PanelOLS
import matplotlib.gridspec as gridspec
from statsmodels.regression.linear_model import OLS

def settings_plot(ax):
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

sizeOfFont=18
ticks_font = font_manager.FontProperties(size=sizeOfFont)
sizefigs_L=(14,8)



path="./Output/"

csv_fname='perplexity_fullsample_'

perp_table=pd.DataFrame()
for k in range(2,101):
    temp=pd.read_csv(path+csv_fname+"%i.csv"%int(k),index_col=0)
    perp_table=perp_table.append(temp,ignore_index=True)

perp_table['Perplexity']=perp_table['Log perplexity'].apply(lambda x: np.exp(-x))

gs = gridspec.GridSpec(1, 1)
fig=plt.figure(facecolor='white',figsize=sizefigs_L)
ax=fig.add_subplot(gs[0, 0])
ax=settings_plot(ax)

plt.plot(perp_table['No topics'],perp_table['Perplexity'])
plt.xlabel('Number of topics',fontsize=18)
plt.ylabel('Model perplexity',fontsize=18)
plt.savefig('perplexity_topics.pdf',bbox_inches='tight')
#plt.show()

ntopics=60

nlp_model=models.LdaModel.load('../pretrained_models/'+"lda_%i.model"%ntopics)


topics = nlp_model.show_topics(num_topics=ntopics, formatted=False)


out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(30, 2, figsize=(10,60), sharey=True, dpi=160)
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color='b', width=0.2, label='Weights')
    ax.set_ylabel('Importance', color='b')
    ax.set_title('Topic: ' + str(i), color='b', fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.savefig('topics_terms_n=%i.pdf'%int(ntopics),bbox_inches='tight')