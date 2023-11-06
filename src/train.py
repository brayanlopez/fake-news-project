# TODO: refactor this file to adapt it for this project

import numpy as np
import pandas as pd
import logging
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import GradientBoostingRegressor

from utils import save_model, save_simple_metrics_report, get_model_performance_test_set

import nltk
nltk.download('stopwords')

stopword_es = nltk.corpus.stopwords.words('spanish')

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading Data...')
data = pd.read_csv('../data/full_data.csv')

logger.info('Loading model pipeline...')
clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopword_es)),
    ('classifier', MultinomialNB())
])

logger.info('Seraparating dataset into train and test')
X_train, X_test, y_train, y_test = train_test_split(
    data["Text"], data.label, test_size=0.33, random_state=53)


logger.info('Setting Hyperparameter to tune')
param_grid = {'tfidf__max_df': np.arange(0, 1, 0.5)}
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')


logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_


logger.info('Cross validating with best model...')
cross_val_scores = cross_val_score(
    best_model, X_train, y_train, cv=5, scoring='accuracy')

test_score = np.mean(cross_val_scores)

assert test_score > 0.65

logger.info(f'Cross validations scores: {cross_val_scores}')
logger.info(f'Mean cross validations scores: {cross_val_scores.mean()}')

logger.info('Updating model...')
save_model(best_model)

logger.info('Generating model report...')
# TODO: generate model report
# validation_score = best_model.score(X_test, y_test)
# save_simple_metrics_report(train_score, test_score, validation_score, best_model)

# y_test_pred = best_model.predict(X_test)
# get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')
