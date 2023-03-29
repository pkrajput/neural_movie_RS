# Movie Recommendations using NeuralCF and DeepFM for Warm Start Scenario
Recommendation System Final Project; Skoltech 2023

## Dataset and Model

#### Dataset

We used data from the Kaggle competition: Skoltech Recommendation System Challenge 2023, which provides two types of datasets: training and testset. There are four columns: `userid` and `itemid` columns, which correspond to int64 userIDs and itemIDs, respectively, `timestamp` column, which corresponds to int64 unix timestamps, and `rating` column, which has float64 values. The rating scale has 0.5 increments ranging from 0.5 to 5.0. 

- Training: there are 127,282 userIDs and 18,264 movieIDs.
- Testset: there are 2,963 userIDs and 17,102 movieIDs.

The dataset can be downloaded by [link for download dataset](https://www.kaggle.com/competitions/skoltech-recommender-systems-challenge-2023/data?select=training).


#### Model

To predict the top 20 recommended movies, we use Collaborative Full Feedback model (CoFFee) [(Frolov & Oseledets, 2016)](https://arxiv.org/abs/1607.04228). This is a tensor-based method that can be used in a warm-start scenario. The data can be represented as a tensor with three dimensions: users, items, and ratings. This tensor will be Tucker decomposed and can be represented as:

$$ 
\mathcal{R} = \mathcal{G} \times_1 U \times_2 V \times_3 W. 
$$

Higher order folding-in will be used in the method to predict recommendations. If $P$ is is a new user preference matrix, the data can be updated to the Tucker model and can computed:

$$ 
R \approx VV^{T}PWW^{T}.
$$

DeepFM and NeuralCF are two popular models for developing recommendation systems. To learn user and item embeddings, NeuralCF employs a neural network. This enables the model to capture user-item relationships and make personalized recommendations.

DeepFM enhances recommendation performance by combining factorization machines and neural networks. Factorization machines are a popular technique for modeling the interactions between features such as user preferences and item attributes in recommendation systems.



## Step-by-step to Building a Movie Recommendation System

#### 1. Data Collection and Pre-processing
To begin, we should collect the training and testset data from Kaggle by clicking on [this link](https://www.kaggle.com/competitions/skoltech-recommender-systems-challenge-2023/data?select=training).

When using ```pd.read csv``` for training data, we must remove the last row that contains a ```NaN``` value. We must preprocess the data by usinf function from `dataprep.py` before training it.
We calculate the average, standard deviation, and rating for each user in the `deepFM train.ipynb` script. 

#### 2. Model Training 
Let us begin training the model. We use `neuralCF train.ipynb` and `deepFM.ipynb` to train the neuralCF and deepFM, respectively, and save the trained model.

#### 3. Getting Recommendation
- `folding_in.ipynb`: Following the seminar in class, we used the Collaborative Full Feedback model (CoFFee) to decompose the tensor that contained userIDs, movieIDs, and ratings. After obtaining useful information from the training data, we apply the testset to the model to obtain the top-50 recommended movies by using function `evaluation.py` from the seminar and save the result.
- `top_rec_20_update.ipynb`: After running `folding_in.ipynb` and obtaining the result, we run neuralCF to obtain the top-100 recommended movies and concatenate the recommended movies without duplicated movieIDs. Following that, we use the information from these recommended movies to feed into the deepFM model, obtaining the final top-20 recommended movies.

#### 4. Testing
We check the model score after receiving the top-20 movie recommendations by submitting the result to [Kaggle](https://www.kaggle.com/competitions/skoltech-recommender-systems-challenge-2023/leaderboard). **NDCG** is the evaluation metric used in Kaggle to measure the quality of ranking.


## References
Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient methods for online learning and stochastic optimization. 2011. doi: [https://doi.org/10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980).

Frolov, E. and Oseledets, I. Fifty shades of ratings: How to benefit from a negative feedback in top-n recommendations tasks. 2016. doi: [http://dx.doi.org/10.1145/2959100.2959170](http://dx.doi.org/10.1145/2959100.2959170).

Guo, H., Ye, T. Y., Li, Z., He, X., and Dong, Z. Deepfm: An end-to-end wide deep learning framework for ctr prediction. 2018. doi: [https://doi.org/10.48550/arXiv.1804.04950](https://doi.org/10.48550/arXiv.1804.04950).

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S. Neural collaborative filtering. 2017. doi: [https://doi.org/10.48550/arXiv.1708.05031](https://doi.org/10.48550/arXiv.1708.05031).

Kingma, D. P. and Ba, J. L. Adam: A method for stochastic optimization. 2017. doi: [https://doi.org/10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980).

Kurbiel, T. and Khaleghian, S. Training of deep neural networks based on distance measures using rmsprop. 2017. doi: [https://doi.org/10.48550/arXiv.1708.01911](https://doi.org/10.48550/arXiv.1708.01911).

Rabanser, S., Shchur, O., and Gunnemann, S. Introduction to tensor decompositions and their applications in machine laerning. 2017. doi: [https://doi.org/10.48550/arXiv.1711.10781](https://doi.org/10.48550/arXiv.1711.10781).

Schafer, J. B., Frankowski, D., Herlocker, J., and Sen, S. Collaborative filtering recommender systems. 2007.




