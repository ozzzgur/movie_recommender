import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ready_df = pd.read_csv("ready_df.csv")
combined_nodes= pd.read_csv("combined_nodes.csv")
route= combined_nodes["Node"].tolist()
node1= route[0]
node2= route[-1]
plt.figure(figsize=(10, 8))

# Create the scatter plot with Seaborn, using different colors for each genre
sns.scatterplot(data=ready_df, x="X", y="Y", hue="genres1", palette="tab20b")

# Highlight specific nodes (in red and a little bit bigger)
# Highlight nodes between 2 and 94 (in green and a little bit bigger)
sns.scatterplot(data=ready_df.loc[route], x="X", y="Y", color="green", s=100, label="Between Nodes")

target_source = [node1,node2]

sns.scatterplot(data=ready_df.loc[target_source], x="X", y="Y", color="red", s=100, label="Highlighted Nodes")

# Show the legend
plt.legend(title="Genres")

# Show the plot
plt.show()
def check_name(num):
    print(ready_df[ready_df.index==num]["title"])



