from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import keras
from joblib import dump, load
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")

class RecommenderNet(keras.Model):
  def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
    """ Constructor
      :param int num_users: number of users
      :param int num_items: number of items
      :param int embedding_size: the size of embedding vector
    """
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_items = num_items
    self.embedding_size = embedding_size
    
    # Define a user_embedding vector
    # Input dimension is the num_users
    # Output dimension is the embedding size
    self.user_embedding_layer = keras.layers.Embedding(
      input_dim=num_users,
      output_dim=embedding_size,
      name='user_embedding_layer',
      embeddings_initializer="he_normal",
      embeddings_regularizer=keras.regularizers.l2(1e-6),
    )
    # Define a user bias layer
    self.user_bias = keras.layers.Embedding(
      input_dim=num_users,
      output_dim=1,
      name="user_bias"
    )
    # Define an item_embedding vector
    # Input dimension is the num_items
    # Output dimension is the embedding size
    self.item_embedding_layer = keras.layers.Embedding(
      input_dim=num_items,
      output_dim=embedding_size,
      name='item_embedding_layer',
      embeddings_initializer="he_normal",
      embeddings_regularizer=keras.regularizers.l2(1e-6),
    )
    # Define an item bias layer
    self.item_bias = keras.layers.Embedding(
      input_dim=num_items,
      output_dim=1,
      name="item_bias"
    )
      
  def call(self, inputs):
    """
        method to be called during model fitting
        :param inputs: user and item one-hot vectors
    """
    # Compute the user embedding vector
    user_vector = self.user_embedding_layer(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    item_vector = self.item_embedding_layer(inputs[:, 1])
    item_bias = self.item_bias(inputs[:, 1])
    dot_user_item = tf.tensordot(user_vector, item_vector, 2)
    # Add all the components (including bias)
    x = dot_user_item + user_bias + item_bias
    # Sigmoid output layer to output the probability
    return tf.nn.relu(x)


def add_new_ratings(new_courses):
  res_dict = {}
  if len(new_courses) > 0:
    # Create a new user id, max id + 1
    ratings_df = load_ratings()
    new_id = ratings_df['user'].max() + 1
    users = [new_id] * len(new_courses)
    ratings = [3.0] * len(new_courses)
    res_dict['user'] = users
    res_dict['item'] = new_courses
    res_dict['rating'] = ratings
    new_df = pd.DataFrame(res_dict)
    # updated_ratings = pd.concat([ratings_df, new_df])
    new_df.to_csv("user_ratings.csv", index=False)
    return new_id

# Create course id to index and index to id mappings
def get_doc_dicts():
  bow_df = load_bow()
  grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
  idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
  id_idx_dict = {v: k for k, v in idx_id_dict.items()}
  del grouped_df
  return idx_id_dict, id_idx_dict

def get_user_profile(course_genders):
  user_ratings = pd.read_csv("user_ratings.csv")
  user_course_genders = course_genders[course_genders['course'].isin(user_ratings['item'])]
  user_course_genders = pd.merge(user_course_genders, user_ratings[['item', 'rating']], left_on='course', right_on='item', right_index=False)
  user_course_genders.drop(columns='item', inplace=True)
  genders = user_course_genders.columns[1:-1]
  user_course_genders[genders] = user_course_genders[genders].multiply(user_course_genders['rating'], axis='index')
  return normalize(user_course_genders[genders].sum().values[np.newaxis])

# def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
def course_similarity_recommendations(user_id, threshold):
  user_courses = pd.read_csv("user_ratings.csv")['item'].values
  sim = pd.read_csv("sim.csv").set_index('course')
  scores = []
  courses = []
  users = []
  for course in user_courses:
    # Get the similarity with other courses
    scores_ = sim[sim.index == course].values
    # Remove courses already taken by the user
    index_to_delete = np.where(np.in1d(sim.columns.values, user_courses))[0]
    scores_ = np.delete(scores_, index_to_delete)
    courses_ = np.delete(sim.columns.values, index_to_delete)
    # Remove courses with score less than threshold
    index_to_delete = np.where(scores_ < threshold)
    courses_ = np.delete(courses_, index_to_delete)
    scores_ = np.delete(scores_, index_to_delete)
    scores = scores + scores_.tolist()
    courses = courses + courses_.tolist()
  users = [user_id] * len(scores)
  return users, courses, scores

def user_profile_recommendations(user_id):
  # Load the data
  course_genders = pd.read_csv("course_gender.csv")
  user_ratings = pd.read_csv("user_ratings.csv")
  genders = course_genders.columns[1:]
  # Get the user profile
  user_profile = get_user_profile(course_genders)
  # Get unseen courses
  new_course_genders = course_genders[~course_genders['course'].isin(user_ratings['item'])]
  # Calculate the recommendations
  recommendations = np.array(np.dot(user_profile[np.newaxis], new_course_genders[genders].values.T))
  sorted = recommendations.argsort()[0][::-1]
  users = [user_id] * len(new_course_genders)
  courses = new_course_genders['course'].values
  scores = recommendations.take(sorted)
  return users, courses, scores

def clustering_recommendations(user_id, pca: bool):
  # Define needed variables
  user_profiles = pd.read_csv("user_profiles.csv")
  course_genders = pd.read_csv("course_gender.csv")
  ratings = pd.read_csv("ratings.csv")
  # Load the trained model
  if pca:
    model: Pipeline = load('pca_kmeans.joblib')
  else:
    model: Pipeline = load('kmeans.joblib')
  # Get the user profile
  user_profile = get_user_profile(course_genders)
  users = user_profiles['user'].values
  # Get the cluster that the user belongs to
  user_cluster = pd.DataFrame({'user': users, 'cluster': model['kmeans'].labels_})
  # Get the other users in the same cluster
  users_related = user_cluster[user_cluster['cluster'] == model.predict(user_profile[np.newaxis])[0]]['user'].values
  # Get the courses already taken by the user
  user_courses = ratings[ratings['user'] == user_id]['item'].values
  # Get the courses unseen by the user which other users in the same cluster already finish
  recommendations: pd.Series = ratings[
    (ratings['user'].isin(users_related)) &
    (~ratings['item'].isin(user_courses))
  ]['item'].value_counts()
  users = [user_id] * len(recommendations)
  courses = recommendations.index
  scores = recommendations.values
  return users, courses, scores

def knn_recommendations(user_id):
  # Get the all the courses as interactions.columns
  interactions = pd.read_csv("interactions.csv").set_index('user')
  courses = interactions.columns
  # Get the user interactions
  user_ratings = pd.read_csv("user_ratings.csv")
  user_interactions = user_ratings.pivot(index='user', columns='item', values='rating').fillna(0).reset_index(drop=True)
  user_interactions = pd.concat([pd.DataFrame(columns=interactions.columns), user_interactions]).fillna(0)
  user_courses = user_ratings['item'].unique()
  # Load the KNN model
  knn: NearestNeighbors = load("knn.joblib")
  # Predict the closest neighbors and it's distances
  distances, indices = knn.kneighbors(user_interactions.iloc[0].values[np.newaxis], return_distance=True)
  distances = distances[0]
  indices = indices[0]
  neighbors_interaction = interactions.iloc[indices].values
  neighbors_similarity = 1 - distances[np.newaxis]
  # Calculate the recommendations
  scores: np.array = (np.dot(neighbors_similarity, neighbors_interaction)/neighbors_similarity.sum())[0]
  scores[np.argwhere(np.isnan(scores))] = 0
  # Keep only the courses not taken by the user
  index_to_delete = np.where(np.in1d(courses, user_courses))[0]
  courses = np.delete(courses, index_to_delete)
  scores = np.delete(scores, index_to_delete)
  # Remove courses with score of 0
  index_to_delete = np.where(scores == 0)
  courses = np.delete(courses, index_to_delete)
  scores = np.delete(scores, index_to_delete)
  # Sort results
  sorted_indexes = np.argsort(scores)[::-1]
  courses = courses[sorted_indexes]
  scores = scores[sorted_indexes]
  users = [user_id] * len(courses)
  return users, courses, scores

def nmf_recommendations(user_id):
  # Get the all the courses as interactions.columns
  interactions = pd.read_csv("interactions.csv")
  interactions.drop(columns='user', inplace=True)
  courses = interactions.columns
  # Get the user interactions
  user_ratings = pd.read_csv("user_ratings.csv")
  user_interactions = user_ratings.pivot(index='user', columns='item', values='rating').fillna(0).reset_index(drop=True)
  user_interactions = pd.concat([pd.DataFrame(columns=interactions.columns), user_interactions]).fillna(0)
  user_courses = user_ratings['item'].unique()
  # Load de NMF model
  nmf: NMF = load("nmf.joblib")
  # Calculate the recommendations
  new_user_transformed = nmf.transform(user_interactions)
  scores = np.dot(new_user_transformed, nmf.components_)[0]
  # Remove the courses already taken by the user
  index_to_delete = np.where(np.in1d(courses, user_courses))
  courses = np.delete(courses, index_to_delete)
  scores = np.delete(scores, index_to_delete)
  # Remove courses with score of 0
  index_to_delete = np.where(scores == 0)
  courses = np.delete(courses, index_to_delete)
  scores = np.delete(scores, index_to_delete)
  # Sort results
  sorted_indexes = np.argsort(scores)[::-1]
  courses = courses[sorted_indexes]
  scores = scores[sorted_indexes]
  users = [user_id] * len(courses)
  return users, courses, scores

# Model training
def train(model_name, params):
  if model_name == "Clustering":
    user_profiles = pd.read_csv("user_profiles.csv")
    features = user_profiles.drop(columns='user')
    kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=params['n_clusters']))])
    kmeans.fit(features)
    dump(kmeans, "kmeans.joblib")
  elif model_name == "Clustering with PCA":
    user_profiles = pd.read_csv("user_profiles.csv")
    features = user_profiles.drop(columns='user')
    pca_kmeans = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=params['n_pca'])), ('kmeans', KMeans(n_clusters=params['n_clusters']))])
    pca_kmeans.fit(features)
    dump(pca_kmeans, "pca_kmeans.joblib")
  elif model_name == "KNN":
    ratings = pd.read_csv("ratings.csv")
    interactions = ratings.pivot(index='user', columns='item', values='rating').fillna(0).reset_index(drop=True)
    knn = NearestNeighbors(metric="cosine", n_neighbors=params['n_neighbors'])
    knn.fit(interactions)
    dump(knn, "knn.joblib")
  elif model_name == "NMF":
    interactions = pd.read_csv("interactions.csv").drop(columns='user')
    nmf = NMF(n_components=params['n_components'])
    nmf.fit(interactions)
    dump(nmf, "nmf.joblib")
    
# Prediction
def predict(model_name, user_ids, params) -> pd.DataFrame:
  # Variables for all models
  users = []
  courses = []
  scores = []
  res_dict = {}
  if model_name == "Course Similarity":
    for user_id in user_ids:
      users, courses, scores = course_similarity_recommendations(user_id, params["sim_threshold"] / 100.0)
  elif model_name == "User Profile":
    for user_id in user_ids:
      users, courses, scores = user_profile_recommendations(user_id)
  elif model_name == "Clustering":
    for user_id in user_ids:
      users, courses, scores = clustering_recommendations(user_id, False)
  elif model_name == "Clustering with PCA":
    for user_id in user_ids:
      users, courses, scores = clustering_recommendations(user_id, True)
    pass
  elif model_name == "KNN":
    for user_id in user_ids:
      users, courses, scores = knn_recommendations(user_id)
  elif model_name == "NMF":
    for user_id in user_ids:
      users, courses, scores = nmf_recommendations(user_id)
  res_dict['USER'] = users
  res_dict['COURSE_ID'] = courses
  res_dict['SCORE'] = scores
  res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
  res_df = res_df.groupby(by=['USER', 'COURSE_ID']).mean().reset_index().sort_values(by='SCORE', ascending=False)
  return res_df.head(params['top_courses'])