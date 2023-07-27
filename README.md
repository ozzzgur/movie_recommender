# Movie Content Recommender System

This project demonstrates the development of a sneaky movie content recommender system that gently guides a user from an action movie, such as "Terminator 2," to a drama movie, "Cinderella," using a content-based approach. The recommendation system leverages techniques such as TF-IDF vectorization, cosine similarity, and Euclidean distances to establish correlations between movies based on their content and descriptions.

## Data Preparation (data_preparation.py)

In the data preparation phase, we perform the following steps:

1. Importing essential libraries, including pandas for data manipulation and ast for handling movie genre data in JSON format.
2. Parsing the movie genres from JSON format to make them more accessible.
3. Creating a TF-IDF vectorizer to convert movie descriptions into numerical representations, reducing their size using PCA (Principal Component Analysis) to determine X-Y coordinates.
4. Calculating pairwise Euclidean distances and cosine similarities between all movies based on their TF-IDF representations.
5. Saving the results as CSV files for further analysis.

## Tuning (threshold_func.py)

In the tuning phase, we use the data saved in CSV files to determine the path between two nodes: "Terminator 2" and "Cinderella." We perform the following steps:

1. Calculating the distance between the two nodes and selecting nearby nodes based on a threshold_distance function.
2. Searching for nodes with similar properties to these two points using a threshold_cosine function.
3. Finding the intersection of the nodes obtained from the previous functions and saving them as "combined_nodes.csv".

## Workflow Overview

1. Data Preparation: The initial step involves processing the dataset and creating TF-IDF representations for movie descriptions. Euclidean distances and cosine similarities are computed and saved as CSV files.

2. Tuning: The threshold functions are applied to determine points between "Terminator 2" and "Cinderella." These points act as recommendations and will guide the user from action movies to drama movies.

## Medium Article

For a detailed walkthrough of the project and implementation, you can check out the accompanying Medium article at [Building a Sneaky Movie Recommender by Using sklearn.metrics in Python](https://medium.com/@ozzgur-sanli/building-a-sneaky-movie-recommender-by-using-sklearn-metrics-in-python-ef546623dfba).

## Note

This version of the content-based directed recommender system is the first iteration and has been designed to avoid the complexity of mix-integer programming, which would be required for a larger number of nodes. The system provides a smooth transition from one movie genre to another.

Feel free to explore and experiment with this recommender system. If you have any questions or feedback, please don't hesitate to reach out. Enjoy the experience!

[Combined Version of Codes](link_to_combined_version_of_codes)

---

You can use this markdown as part of your GitHub `readme.md` file to describe the project and its implementation. Make sure to replace `link_to_combined_version_of_codes` with the actual link to the combined version of your codes on GitHub.
