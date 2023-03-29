# Movie Recommendations using NeuralCF and DeepFM for Warm Start Scenario
Recommendation System Final project; Skoltech 2023

For movie recommendation, we use CoFFee (Collaborative Full Feedback model) [(Frolov & Oseledets, 2016)](https://arxiv.org/abs/1607.04228) to predict top 20 recommended movies. This is a tensor base method that can apply to warm start scenerio. The data can be construct as tensor which has 3 dimensions: users, items, and ratings. This tensor will be decomposed by Tucker decomposition and can be represent as

$$ 
\mathcal{R} = \mathcal{G} \times_1 U \times_2 V \times_3 W 
$$

For predicting recommendation, higher order folding-in will be apply to the method. Let $P$ is a matrix of a new user preferences, the data can be updated to the Tucker model and can compute:

$$ 
R \approx VV^{T}PWW^{T} 
$$

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






