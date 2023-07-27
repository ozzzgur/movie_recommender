import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from sklearn.decomposition import PCA


df = pd.read_csv("movies_metadata.csv", low_memory=False)
df = df[0:1000]
node1= 582
node2=993

# Convert string representation of lists to actual lists of dictionaries
df["genres"] = df["genres"].apply(ast.literal_eval)

# Create a new column "genres1" containing the first genre names
df["genres1"] = df["genres"].apply(lambda x: x[0]["name"] if x else None)


tfidf = TfidfVectorizer(stop_words="english")
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])

pca = PCA(n_components=2,random_state=41)
coordinates = pca.fit_transform(tfidf_matrix.toarray())
coordinates_df = pd.DataFrame(coordinates, columns=['X', 'Y'])

euclidean_distances = pairwise_distances(tfidf_matrix, metric='euclidean')
euclidean_distances_df = pd.DataFrame(euclidean_distances)
euclidean_distances_df.to_csv('distance.csv', index=False)
cosine_similarity_matrix = cosine_similarity(tfidf_matrix,tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_similarity_matrix)
cosine_sim_df.to_csv("cosine.csv",index=False)

ready_df = pd.concat([df[["title","genres1"]], coordinates_df], axis=1)
ready_df["genres1"] = pd.Categorical(ready_df["genres1"])
ready_df["dist_1"] = euclidean_distances_df.iloc[:,node1]
ready_df["dist_2"] = euclidean_distances_df.iloc[:,node2]


ready_df.to_csv("ready_df.csv", index=False)