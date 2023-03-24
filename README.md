# Adding Contextual Information and User Biases for Neural Network Movies Recommendations

neural RS based on cosine plot similarity and top rating retreival from NeuralCF using DeepFM

The first approach uses cosine plot similarity to recommend movies to users. This approach is based on the idea that movies with similar plot summaries are likely to be enjoyed by the same users. To implement this approach, we first calculate the cosine similarity between the plot summaries of all pairs of movies in the dataset. We then use this similarity matrix to recommend movies to users based on the movies they have already rated. Specifically, we use the similarity matrix to find movies that are similar to the movies the user has rated highly but that the user has not yet seen.

The second approach uses top rating retrieval using NeuralCF and DeepFM. These are neural network models that can predict movie ratings based on user and movie data. NeuralCF is a collaborative filtering model that uses a neural network to learn user and item embeddings that are used to predict ratings. DeepFM is a deep factorization machine model that combines factorization machines with neural networks to improve recommendation performance.



## Dataset and Model

The MovieLens100k dataset contains 100,000 ratings from 943 users on 1,682 movies. The ratings are on a scale from 1 to 5, and there are no missing values in the dataset. In addition to the ratings, the dataset also contains user information such as age, gender, and occupation, as well as movie information such as title, release date, and genres.

In addition to the MovieLens100k dataset, you can also download a trained model that has been trained on the dataset. The trained model is a neural network model that uses collaborative filtering to recommend movies to users.

The model has been trained using Keras, which is a popular deep learning library in Python. The model takes in user and movie data as input and outputs a rating prediction. The model has been trained on the MovieLens100k dataset using a variety of hyperparameters and has achieved a decent accuracy in predicting movie ratings.

You can download dataset, full data, and trained model [here](https://drive.google.com/drive/u/0/folders/1qaKKdpzv0RGDo1OhWe5T8S1GWcWnLOWO).



## Step-by-step Guide to Building a Movie Recommendation System

#### 1. Data Collection
To get started, you need to collect data from IMDb. You can use the `get_imdb_data.py script` to collect the necessary data. 
This script will extract movie titles, genres, and ratings from the IMDb website.

#### 2. Data Pre-processing
Once you have collected the data, you need to preprocess it. The `feature_ext.py` script can be used to select the top two genres, calculate the average, 
standard deviation, and count by user and item. It also finds the user's favorite genre and favorite movieId.

#### 3. Simple Model
Now that you have preprocessed the data, you can build a simple matrix factorization retrieval model using neuralCF. 
This model can be implemented using the `model_trfs.ipynb` notebook.

#### 4. Applied Models
If you want to build more complex models, you can use deep learning models such as neuralCF and DeepFM. 
These models can be implemented using the `model_retreival_ans_ranking.ipynb` notebook.

#### 5. Testing
Once you have built the models, you need to test and compare their efficiency. This can be done using the `get_top_5_recommendation.ipynb` notebook.






