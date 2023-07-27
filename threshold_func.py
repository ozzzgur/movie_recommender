import pandas as pd
ready_df = pd.read_csv("ready_df.csv")
distance = pd.read_csv("distance.csv")
cosine = pd.read_csv("cosine.csv")
node1= 582
node2=993

def threshold_distance(threshold=1.8):
  # Find the nodes that lie in between 2 and 94
    between_nodes = ready_df[(ready_df['dist_1'] <= threshold) & (ready_df['dist_2'] <= threshold)]
    return between_nodes.index

def threshold_cosine(threshold_cosine_similarity=0.02):
    cosine_array = cosine.to_numpy()
    between_nodes_indices_cosine = [i for i, sim in enumerate(cosine_array[node1]) if 0 < sim < threshold_cosine_similarity]
    between_nodes_indices_cosine.extend([i for i, sim in enumerate(cosine_array[node2]) if 0 < sim < threshold_cosine_similarity])
    between_nodes_indices_cosine = list(set(between_nodes_indices_cosine))
    return between_nodes_indices_cosine

def find_intersection_with_thresholds(max_intersection=5):
    distance_threshold = 1.42 #Tuning Parameters
    cosine_threshold = 0.05  #Tuning Parameters
    intersection_nodes = []

    while True:
        between_nodes_indices_euclidean = threshold_distance(threshold=distance_threshold)
        between_nodes_indices_cosine = threshold_cosine(threshold_cosine_similarity=cosine_threshold)

        between_nodes_indices_combined = list(
            set(between_nodes_indices_euclidean).intersection(between_nodes_indices_cosine))

        intersection_count = len(between_nodes_indices_combined)

        if intersection_count <= max_intersection:
            intersection_nodes = between_nodes_indices_combined
            break

        if distance_threshold <= 0 and cosine_threshold <= 0:
            break

        distance_threshold -= 0.0001
        cosine_threshold -= 0.0001

    return intersection_nodes, distance_threshold, cosine_threshold


max_intersection = 10
intersection_nodes, final_distance_threshold, final_cosine_threshold = find_intersection_with_thresholds(
    max_intersection=max_intersection)

if len(intersection_nodes) == 0:
    print(
        f"No intersection found for nodes that meet both thresholds with a maximum of {max_intersection} intersections.")
else:
    combined_nodes = list(intersection_nodes)
    combined_nodes.insert(0, node1)
    combined_nodes.append(node2)
    combined_df = pd.DataFrame({'Node': combined_nodes})
    combined_df.to_csv('combined_nodes.csv', index=False)
    print(f"Combined nodes (intersection nodes + node1 + node2) saved to 'combined_nodes.csv'.")
    print(f"Final distance threshold used: {final_distance_threshold}")
    print(f"Final cosine threshold used: {final_cosine_threshold}")





