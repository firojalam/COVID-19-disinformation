import pandas as pd
import json

q = 'q3'

# Bert predictions
bert_predictions_file = '/content/drive/My Drive/post-masters/covid19_infodemic_data - Copy/multiclass_english_splits_features/bert_classified_multiclass/' + q + '_bert_classified.tsv'
bert_with_ids_file = '/content/drive/My Drive/post-masters/covid19_infodemic_data - Copy/multiclass_english_splits_features/bert_classified_multiclass/' + q + '_concatenated_test.tsv'

bert_df = pd.read_csv(bert_predictions_file, sep='\t')
bert_with_ids_df = pd.read_csv(bert_with_ids_file, sep='\t')

labels_count = len(bert_df.columns) // 2
labels_ordered = [value for i, value in enumerate(bert_df.iloc[0].values.tolist()) if i % 2 == 0]
labels_dict = {value:i for i, value in enumerate(labels_ordered)}

bert_df['id'] = bert_with_ids_df['tweet_id']
probability_features = ['confidence_' + str(i + 1) for i in range(labels_count)]

# English features file
english_features_base_file = '/content/drive/My Drive/post-masters/covid19_infodemic_data - Copy/multiclass_english_splits_features/' + q + '/test.'
english_features_files = [english_features_base_file + str(i) + '.tsv' for i in range(10)]
english_features_dfs = [pd.read_csv(file, sep='\t') for file in english_features_files]
english_features_df = pd.concat(english_features_dfs)

columns = [c for c in english_features_df.columns if c not in ['text', 'tokenized_text']]
english_features_df = english_features_df[columns]
new_columns = ['id']
new_columns.extend(english_features_df.columns[1:])

english_features_df.columns = new_columns

boolean_columns = [
  'default_profile', 'default_profile_image', 'verified',
  'protected', 'geo_enabled', 'reply', 'quotes', 'contain_media']

for column in boolean_columns:
  english_features_df[column] = english_features_df[column].astype(int)

#print(bert_df.shape, test_set_df.head, english_features_df.head)
bert_tweet_ids = bert_df['id'].tolist()
english_features_tweet_ids = english_features_df['id'].tolist()

def intersection(l1, l2):
  return [elem for elem in l1 if elem in l2]

def str_to_int(str):
  return labels_dict[str]

x_tweet_ids = intersection(bert_tweet_ids, english_features_tweet_ids)

bert_intersection_df = bert_df.loc[bert_df['id'].isin(x_tweet_ids)]
english_features_intersetion_df = english_features_df.loc[english_features_df['id'].isin(x_tweet_ids)]

merged_df = pd.merge(bert_intersection_df, english_features_intersetion_df, on="id")

# Get features
use_new_botometer_features = False

twitter_object_features = [
  'default_profile',
  'default_profile_image', 'verified', 'protected', 'geo_enabled',
  'reply', 'quotes', 'contain_media', 'statuses_count', 'followers_count',
  'friends_count', 'favourites_count', 'listed_count',
  'year_user_created', 'media_type_No media', 'media_type_photo',
  'media_type_video', 'media_type_animated_gif',
  'num_media_0', 'num_media_1', 'num_media_2', 'num_media_3',
  'num_media_4', 'fact_UNKNOWN', 'fact_LOW', 'fact_MIXED', 'fact_HIGH',
  'source_Twitter for iPhone', 'source_Twitter for Android',
  'source_Twitter Web App', 'source_Twitter Media Studio',
  'source_TweetDeck', 'source_Periscope', 'source_TheWhiteHouse',
  'source_Echofon', 'source_Twitter for iPad', 'source_Sprout Social',
  'source_Twitter Web Client', 'source_SocialPilot.co',
  'source_PTI_Tweets', 'source_SocialFlow', 'source_Hootsuite Inc.',
]

tanbih_features = ['tanbih_propoganda', 'tanbih_propoganda_sentences']

botometer_features = [
  'english', 'universal', 'content', 'friend', 'network', 'sentiment', 'temporal', 'user'              
]

bert_features = probability_features

# Modify this to get different feature subsets
x_feature_names = bert_features

x_feature_names.extend(twitter_object_features)
x_feature_names.extend(tanbih_features)
#x_feature_names.extend(botometer_features)

print(len(x_feature_names))

x_features_df = merged_df[x_feature_names]
x_features = x_features_df.values.tolist()

labels = [labels_dict[label] for label in merged_df[q + '_label'].tolist()]

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import uniform, gamma, triang, randint
import xgboost as xgb

n_iter = 200

def get_cv_train_val_test(array, val_index, test_index):
  train = np.concatenate([array[i] for i in range(10) if i != val_index and i != test_index])
  val = np.array(array[val_index])
  test = np.array(array[test_index])

  return train, val, test

def train_xg_boost_with_randomized_search_cv(x, y):
  params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
  }

  data_split_x = np.array_split(x, 10)
  data_split_y = np.array_split(y, 10)

  best_val_f1 = 0
  test_f1_from_best_val = 0

  for i in range(n_iter):
    if i % 25 == 0:
      print('Currently on iteration number ' + str(i))

    param_samples = { key:params[key].rvs(size=1)[0] for key in params }

    val_f1 = [0] * 10
    test_f1 = [0] * 10

    for split in range(10):
      train_x, val_x, test_x = get_cv_train_val_test(data_split_x, split, (split + 1) % 10)
      train_y, val_y, test_y = get_cv_train_val_test(data_split_y, split, (split + 1) % 10)

      xgb_model = xgb.XGBClassifier(**param_samples)
      xgb_model.fit(train_x, train_y)

      val_predicted_y = xgb_model.predict(val_x)
      val_f1[split] = f1_score(val_y, val_predicted_y, average='weighted')

      test_predicted_y = xgb_model.predict(test_x)
      test_f1[split] = f1_score(test_y, test_predicted_y, average='weighted')

    if best_val_f1 < np.mean(val_f1):
      best_val_f1 = np.mean(val_f1)
      test_f1_from_best_val = np.mean(test_f1)

  return test_f1_from_best_val

import numpy as np

print(len(x_features[0]))
test_f1 = train_xg_boost_with_randomized_search_cv(np.array(x_features), np.array(labels))

print('Test F1 weighted:', test_f1)