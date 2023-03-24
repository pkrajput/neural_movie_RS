# neural_movie_RS
neural RS based on cosine plot similarity and top rating retreival from NeuralCF using DeepFM



# Dataset and trained model
Dataset: MovieLens100k

You can download dataset, full data, and trained model [here](https://drive.google.com/drive/u/0/folders/1qaKKdpzv0RGDo1OhWe5T8S1GWcWnLOWO).

# Step-by-step Guide to Building a Movie Recommendation System

## 1. Data Collection
To get started, you need to collect data from IMDb. You can use the `get_imdb_data.py script` to collect the necessary data. 
This script will extract movie titles, genres, and ratings from the IMDb website.

## 2. Data Pre-processing
Once you have collected the data, you need to preprocess it. The `feature_ext.py` script can be used to select the top two genres, calculate the average, 
standard deviation, and count by user and item. It also finds the user's favorite genre and favorite movieId.

## 3. Simple Model
Now that you have preprocessed the data, you can build a simple matrix factorization retrieval model using neuralCF. 
This model can be implemented using the `model_retrieval.ipynb` notebook.

## 4. Applied Models
If you want to build more complex models, you can use deep learning models such as neuralCF and DeepFM. 
These models can be implemented using the `model_ranking.ipynb` notebook.

## 5. Testing
Once you have built the models, you need to test and compare their efficiency. This can be done using the `model_pipeline.ipynb` notebook.






