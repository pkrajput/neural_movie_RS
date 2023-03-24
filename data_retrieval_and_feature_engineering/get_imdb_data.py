import pandas as pd
import pickle
from imdb import Cinemagoer
import re
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
tqdm.pandas()

#--------- apply movie id ---------
def get_movie_id(movie_title, re_ex = False):
    try:
        if re_ex:
            movie_id = ia.search_movie(re.search("\((.*)",movie_title)[0])[0].getID() 
        else:
            movie_id = ia.search_movie(movie_title)[0].getID() 
        # print(movie_title, movie_id)
        return movie_id 
    except:
        return None

# ---------- get movie data from IMDb ----------
def get_movie_data_from_title(movie_title):
    try:
        movie_id = get_movie_id(movie_title, re_ex = False)     
        movie_detail = ia.get_movie(movie_id)   
        movie_rating = movie_detail['rating']
        movie_plot = movie_detail['plot'][0]  
        movie_encoded_plot = model.encode(movie_plot)
        get_movie_data = pd.Series([movie_id, movie_rating, movie_plot, movie_encoded_plot]) 
        # print(movie_id, movie_title, movie_rating, movie_plot, movie_encoded_plot)
        return get_movie_data
    except:
        none_movie_data = pd.Series([None, None, None, None])
        return none_movie_data
    
def re_get_movie_data_from_title(movie_title):
    try:
        movie_id = get_movie_id(movie_title, re_ex = True)     
        movie_detail = ia.get_movie(movie_id)   
        movie_rating = movie_detail['rating']
        movie_plot = movie_detail['plot'][0]  
        movie_encoded_plot = model.encode(movie_plot)
        get_movie_data = pd.Series([movie_id, movie_rating, movie_plot, movie_encoded_plot]) 
        return get_movie_data
    except:
        none_movie_data = pd.Series([None, None,None,None])
        return none_movie_data

# ---------- save pickle file ---------
def save_pk_file(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)  

#--------------------------------------------- main --------------------------------------------------
if __name__ == "__main__":

    model = SentenceTransformer('all-MiniLM-L6-v2')
    ia = Cinemagoer()
    movie = pd.read_csv("u.item", 
                        sep = '|', 
                        header = None, 
                        encoding='latin-1', 
                        names = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDbURL' , 
                                 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 
                                 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                                 'Sci-Fi', 'Thriller', 'War', 'Western']
                                 )

    movie[['imdb_id', 'rating', 'plot summary', 'plot embedding']] = movie.title.progress_apply(get_movie_data_from_title)
    
    # check lost movie
    movie_lost = movie[movie['rating'].isna()]
    movie_lost[['imdb_id', 'rating', 'plot summary', 'plot embedding']] = movie_lost.title.progress_apply(re_get_movie_data_from_title)
    movie_lost_count = movie_lost['rating'].isnull().sum()
    # print(movie_lost_count)

    movie.iloc[movie_lost.index] = movie_lost
    movie_still_lost = movie_lost[movie_lost['rating'].isna()]
    movie.drop(movie_still_lost.index, inplace = True, errors = 'ignore')

    # data from user and merge files
    user_rating = pd.read_csv('u.data', sep = '\t', header = None, names = ["userId", "movieId", "rating", "timestamp"])
    user_rating.drop(user_rating[user_rating['movieId'].isin(movie_still_lost['movieId'])].index, inplace = True)
    user = pd.read_csv('u.user', sep = '|', header = None, names = ['userId', 'age', 'gender', 'occupation', 'zip code'])
    pck_file = user_rating.merge(movie, on = ['movieId']).merge(user, on = ['userId'])

    # save rating_movie_user.pickle
    save_pk_file(pck_file, "rating_movie_user.pickle")

