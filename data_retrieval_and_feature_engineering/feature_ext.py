import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm

tqdm.pandas()

# ---------- save and load pickle file ---------
def save_pk_file(file, filename):
    with open(filename, "wb") as f:
        pickle.dump(file, f)


def load_pk_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


# ---------- select top 2 genres ----------
def select_genres(columns):
    idx = np.where(columns == 1)
    top_genre = pd.Series(
        [
            all_genre[idx[0][0]] if len(idx[0]) > 0 else None,
            all_genre[idx[0][1]] if len(idx[0]) > 1 else None,
        ]
    )
    return top_genre


# --------------------------------------------- main --------------------------------------------------

if __name__ == "__main__":

    data = load_pk_file("rating_movie_user.pickle")

    data["gender"] = data["gender"].apply(lambda x: 0 if x == "F" else 1)  
    data["zipcode_bucket"] = data["zip code"].apply(lambda x: str(x)[:2])  
    data["release_year"] = pd.DatetimeIndex(data["release_date"]).year 

    # collect 2 genres
    all_genre = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
    ]
    data[["movie_genre_1", "movie_genre_2"]] = data[all_genre].progress_apply(select_genres, axis = 1)

    # we have 2 ratings --> from data and imdb
    data["IMDb_rating"] = data["rating_y"]
    data["rating"] = data["rating_x"]
    
	# ------------------------------- get avg, std, count by user and item ---------------------------------------------
	# get avg by user
    data_1 = data.sort_values(["userId", "timestamp"], 
                              ascending=[True, True]).reset_index()
    avg_rating = (data_1.groupby(data_1.userId)["rating"].expanding().mean().reset_index())
    data_1["user_avg_rating"] = avg_rating["rating"]
    
	# get svd by user
    std_rating = (data_1.groupby(data_1.userId)["rating"].expanding().std().reset_index())
    data_1["user_std_rating"] = std_rating["rating"].fillna(0)
    
	# get count by user
    count_rating = (data_1.groupby(data_1.userId)["rating"].expanding().count().reset_index())
    data_1["user_rating_count"] = count_rating["rating"]

    # get avg by item
    data_2 = data_1.sort_values(["movieId", "timestamp"], 
                                ascending=[True, True]).reset_index()
    avg_rating_item = (data_2.groupby(data_2.movieId)["rating"].expanding().mean().reset_index())
    data_2["movie_avg_rating"] = avg_rating_item["rating"]
    
	# get std by item
    std_rating_item = (data_2.groupby(data_2.movieId)["rating"].expanding().std().reset_index())
    data_2["movie_std_rating"] = std_rating_item["rating"].fillna(0)
    
	# get count by item
    count_rating_item = (data_2.groupby(data_2.movieId)["rating"].expanding().count().reset_index())
    data_2["movie_rating_count"] = count_rating_item["rating"]
    # ---------------------------------------------------------------------------------------------------------------

    data_3 = data_2.copy()
    data_3 = data_3.sort_values(["userId", "timestamp"], ascending=[True, True]).reset_index(drop=True)

    # rating >= 4 --> sum
    for genre in all_genre:
        fav = (
            data_3.groupby(data_3.userId)
            .apply(lambda x: x[x["rating_x"] >= 4])[genre]
            .expanding()
            .sum()
            .reset_index()
        )
        data_3[genre + "_sum"] = fav[genre]

    movie_genres_sum = [genre + "_sum" for genre in all_genre]
    data_3["user_fav_genre"] = data_3[movie_genres_sum].idxmax(axis = 1)
    data_3["user_fav_genre"] = data_3["user_fav_genre"].str.replace("_sum", "")

    data_4 = data_3.copy()
    data_4["rating_and_movie"] = data_4["rating"].map(str) + data_4["movieId"].map(str)
    data_4["rating_and_movie"] = data_4["rating_and_movie"].astype(int)

    temp1 = (
        data_4.groupby(data_4.userId)["rating_and_movie"]
        .expanding()
        .max()
        .reset_index()
    )

    data_4["user_fav_movieId"] = (temp1["rating_and_movie"].astype(int).map(str).str[1:].astype(int))

    all_data = data_4[
        [
            "userId",
            "age",
            "gender",
            "occupation",
            "zipcode_bucket",  
            "movieId",
            "imdb_id",
            "title",
            "movie_genre_1",
            "movie_genre_2",
            "IMDb_rating", 
            "plot embedding",
            "release_year", 
            "rating",
            "user_avg_rating",
            "user_std_rating",
            "user_rating_count", 
            "movie_avg_rating",
            "movie_std_rating",
            "movie_rating_count",  
            "user_fav_genre",
            "user_fav_movieId",
            "timestamp",  
        ]
    ]
    save_pk_file(all_data, "data_2.pickle")
