import pandas as pd
import ast
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("movies_metadata.csv", low_memory=False)
df = df[0:10000]

def check_name(num):
    return (df[df.index==num]["title"])

def check_index(name):
    return (df[df.title==name])

tfidf = TfidfVectorizer(stop_words="english")
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Convert string representation of lists to actual lists of dictionaries
df["genres"] = df["genres"].apply(ast.literal_eval)

# Create a new column "genres1" containing the first genre names
df["genres1"] = df["genres"].apply(lambda x: x[0]["name"] if x else None)

pca = PCA(n_components=2,random_state=41)
coordinates = pca.fit_transform(tfidf_matrix.toarray())
coordinates_df = pd.DataFrame(coordinates, columns=['X', 'Y'])

euclidean_distances = pairwise_distances(tfidf_matrix, metric='euclidean')
euclidean_distances_df = pd.DataFrame(euclidean_distances)
euclidean_distances_df.to_csv('distances.csv', index=False)

ready_df = pd.concat([df[["title","genres1"]], coordinates_df], axis=1)
ready_df["genres1"] = pd.Categorical(ready_df["genres1"])
ready_df.to_csv("ready_df.csv", index=False)

# Calculate the squared Euclidean distances of all nodes from nodes 2 and 94
ready_df['dist_582'] = (ready_df['X'] - ready_df.loc[582, 'X'])**2 + (ready_df['Y'] - ready_df.loc[582, 'Y'])**2
ready_df['dist_7834'] = (ready_df['X'] - ready_df.loc[7834, 'X'])**2 + (ready_df['Y'] - ready_df.loc[7834, 'Y'])**2

# Set a threshold distance to determine which nodes lie in between
threshold = 0.065

# Find the nodes that lie in between 2 and 94
between_nodes = ready_df[(ready_df['dist_582'] <= threshold) & (ready_df['dist_7834'] <= threshold)]
between_nodes_indices = between_nodes.index

# Set the figure size (width, height) in inches
plt.figure(figsize=(10, 8))

# Create the scatter plot with Seaborn, using different colors for each genre
sns.scatterplot(data=ready_df, x="X", y="Y", hue="genres1", palette="tab20b")

# Highlight specific nodes (in red and a little bit bigger)
# Highlight nodes between 2 and 94 (in green and a little bit bigger)
sns.scatterplot(data=ready_df.loc[between_nodes_indices], x="X", y="Y", color="green", s=100, label="Between Nodes")


highlight_indices = [582, 7834]
sns.scatterplot(data=ready_df.loc[highlight_indices], x="X", y="Y", color="red", s=100, label="Highlighted Nodes")

# Show the legend
plt.legend(title="Genres")

# Show the plot
plt.show()
