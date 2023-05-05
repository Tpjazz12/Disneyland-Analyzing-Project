# Disneyland Reviews Analyzing 

An attempt to perform sentiment analysis on Disneyland reviews using various natural language processing (NLP) techniques and machine learning models. This project provides a comprehensive analysis of Disneyland reviews and offers valuable insights into customer satisfaction and experience.

## Overview

* This repository contains an in-depth analysis of Disneyland reviews using various natural language processing (NLP) techniques and machine learning models. The project covers several aspects, including exploratory data analysis (EDA), word clouds, sentiment analysis using VADER, TextBlob, and Hugging Face, and sentiment classification using logistic regression models.

## Summary of Work Done

* The main goal of this project is to analyze Disneyland reviews to gain insights into the sentiments expressed by customers and understand which factors contribute to their experience. The analysis begins with an EDA, which involves cleaning the dataset, removing stop words using the NLTK library, and creating word clouds for each rating category to visualize the most frequently used words in the reviews.

* Next, we perform sentiment analysis using three different approaches: VADER, TextBlob, and Hugging Face. Each approach provides a unique perspective on the reviews' sentiment scores, which can be used to understand the general sentiment trends in the data.

* Finally, we train logistic regression models to classify the reviews into different sentiment categories (Negative, Neutral, Positive) based on the sentiment scores obtained from the VADER, TextBlob, and Hugging Face sentiment analysis. The performance of these models is evaluated using precision, recall, and F1-score metrics, allowing us to compare and identify the best performing approach.

### Data
 * Data:
   * Type: Text data (Disneyland reviews)
   * Input: CSV file containing review text and associated ratings
   * Size: 42,656 reviews
   * Instances (Train, Test Split): 80% for training, 20% for testing
   
 * Preprocessing / Clean up
   * The dataset was cleaned by removing unnecessary characters, punctuations, and stopwords using the NLTK library.  

 * Data Visualization
   * Word clouds were created for each rating category to visualize the most frequently used words in the reviews.
   * Various plots were used during the EDA to understand the data's characteristics, distribution, and trends.

### Problem Formulation
 * Define:
   * Input: Sentiment scores from VADER, TextBlob, and Hugging Face
   * Output: Sentiment category (Negative, Neutral, Positive)
   * Models: Logistic Regression models were used for sentiment classification.
   * Loss, Optimizer, other Hyperparameters: Default settings were used for logistic regression.

### Training
  * Logistic regression models were trained using the Scikit-learn library.
  * Models were trained on a personal computer.
  * Training was relatively quick due to the simplicity of logistic regression.
  * Models were evaluated using precision, recall, and F1-score metrics.
  
### Performance Comparison
  * Key performance metrics: Precision, recall, and F1-score.
  * Results were presented in tables and visualizations to compare the performance of logistic regression models based on VADER, TextBlob, and Hugging Face sentiment scores.
  **VADER and TextBlob:**

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.34      | 0.56   | 0.42     | 720     |
| Neutral   | 0.20      | 0.35   | 0.26     | 1035    |
| Positive  | 0.92      | 0.75   | 0.82     | 6777    |
|           |           |        |          |         |
| Accuracy  |           |        | 0.68     | 8532    |
| Macro avg | 0.49      | 0.55   | 0.50     | 8532    |
| Weighted avg | 0.78   | 0.68   | 0.72     | 8532    |

**Hugging Face:**

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.09      | 0.87   | 0.16     | 669     |
| Neutral   | 0.16      | 0.19   | 0.17     | 943     |
| Positive  | 0.77      | 0.03   | 0.06     | 6393    |
|           |           |        |          |         |
| Accuracy  |           |        | 0.12     | 8005    |
| Macro avg | 0.34      | 0.37   | 0.13     | 8005    |
| Weighted avg | 0.64   | 0.12   | 0.08     | 8005    |

### Conclusions
* From the table, we csn draw to this conclusion:
   * The VADER and TextBlob based model performed significantly better than the Hugging Face based model. The VADER and TextBlob model achieved an accuracy of 0.68, while the Hugging Face model had an accuracy of 0.12.
   * The VADER and TextBlob based model's performance in classifying negative and positive sentiments was quite good, with a precision of 0.34 and 0.92, respectively. However, its performance in identifying neutral sentiment was relatively lower, with a precision of 0.20.
   * The Hugging Face model struggled to classify positive sentiment, as indicated by the low precision and F1-score. This model might not be well-suited for this specific dataset or problem.
   *  The EDA and word clouds provided valuable insights into the dataset, such as the most common words used in each rating category. This information could potentially help improve the sentiment analysis models by incorporating more domain-specific knowledge or refining the feature extraction process.
* The VADER and TextBlob based model performed significantly better for this specific dataset. Future work could involve refining the Hugging Face model, exploring other machine learning algorithms, or incorporating more domain-specific knowledge to further improve sentiment analysis performance.

### Future Work
* Experiment with other classification algorithms such as Random Forest or SVM to improve the sentiment classification.
* Perform a more comprehensive feature engineering process to identify other potentially informative features for sentiment classification.
* Explore more advanced deep learning models, such as BERT, to perform sentiment analysis directly on the text data.

## How to Reproduce Results
* Clone the repository to your local machine or open it in Google Colab.
* Install the required Python packages as listed in the "Getting Started" section.
* Download the dataset and preprocess it as described in the Jupyter Notebook files.
* Train the models as described in the respective Jupyter Notebook files.
* Evaluate the performance of the models and compare the results.

## Overview of Files in Repository
* Sentiment Analysis.ipynb: Vader and Textblob Sentiment Analysis
* Disneyland Reviews Classification.ipynb: 
* EDA and Word Clouds.ipynb: Exploratory data analysis and word cloud generation.
* Huggin Face Sentiment Analysis.ipynb: Hugging Face Transformer models for sentiment analysis.
* Logistic Regression on Hugging Face Sentiment.ipynb: Application of logistic regression on the Hugging Face sentiment analysis results.
* Logistic Regression on VADER and TextBlob.ipynb: Training and testing of logistic regression on VADER and TextBlob feature extraction results.
* More EDA.ipynb: Additional exploratory data analysis

## Software Setup

Required packages include:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* wordcloud
* nltk
* textblob
* vaderSentiment
* transformers (Hugging Face)
* tensorflow

## Training

* We trained two separate logistic regression models on the dataset, one using VADER and TextBlob features, and another using the Hugging Face sentiment analysis features. We used the LogisticRegression class from the sklearn.linear_model package for training both models.
* The training was performed on a standard personal computer, and it was relatively fast, taking only a few minutes to complete. The models were trained using the default settings for loss function and optimizer.
* We decided to stop training when the models converged, and no further improvement was observed.
* There were no significant difficulties faced during training, as logistic regression is a relatively simple and fast algorithm.
* Deep learning models like Hugging Face Transformers took a long time to perform the result. 

## Performance Evaluation
* To evaluate the performance of the two models, we used precision, recall, and F1-score metrics, which were computed using the classification_report function from the sklearn.metrics package.
* The VADER and TextBlob based model achieved an accuracy of 0.68, while the Hugging Face based model had an accuracy of 0.12. The VADER and TextBlob model performed significantly better, particularly in classifying positive sentiment.

## Citation





      






