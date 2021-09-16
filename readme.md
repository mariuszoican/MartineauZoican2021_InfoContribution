## How to run the NLP code?

1. First we train the model using:
    `\train_models\LDA_model_estimation.py`
   or `LDAMarius.slurm` if using a multi-core system.
   * The model files are saves under `.\pretrained_models`
   * Estimation logs are saved under `.\train_models\Logs`
   * Perplexity scores are saved under `.\train_models\Output`
     
2. Second, we run the `\train_models\study_topics.py` file to plot perplexity against 
the number of topics and select the optimal topics.
   * Output is a graph, `perplexity_topics.pdf`.
   * The file also outputs the top ten words for each topic, given a 
     (manually) prespecified number of topics **X**: `topics_terms_n=X.pdf`.
     
3. The file `industry_gettopics.py` generates a quarter-industry panel of topic loadings, saved in the file:
   `.\IndustryAnalysis\topic_loadings_by_industryquarter.csv'`

4. Code `.\IndustryAnalysis\industry_toptopics.py` generates the top 2 topics (with list of words) for each GIC code and saves in `TopTopics_Industries.csv`.
   
4. Code `build_shapley.py` 
   (together with `ShapleyMarius.slurm`) generate panels of Shapley values by analyst-ticker-quarter 
   (including information diversity, contribution), saved in `OutputShapley` folder.

5. Use `merge_shapley.py` in the `OutputShapley` folder to generate a `DataShapley.csv` file.

6. Run `get_technicaldummy.py` to get a file with analyst-level topic loadings on technical analysis topics (`DataShapley_TechnicalTopicWeights.csv')

7. The complete merged file (`DataShapley.csv` + `DataShapley_TechnicalTopicWeights.csv') is saved as `Data_InfoContributionAnalyst.csv' 