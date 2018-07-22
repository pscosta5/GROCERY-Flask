import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from pickle import load

user_products_sentence__prior = pd.read_csv('static/recommender/user_products_sentence__prior(web).csv')
user_products_lookup = pd.read_csv('static/recommender/user_products_lookup.csv').set_index('user_id')

tfidf = joblib.load('static/recommender/tfidf.pkl')
tfidf_matrix = joblib.load('static/recommender/tfidx_matrix.pkl')
product_id_to_name_dict = load(open('static/recommender/product_id_to_name_dict.pickle', 'rb'))

def make_sparse(user_selection_sentence, tfidf_model=tfidf):
    # Input is "sentence" of a string of product ids. First convert to a list of string product ids
    selected_products = (user_selection_sentence.split())
    unique_products = list(set(selected_products))

    # Now make this into a sparse matrix

    # Make matrix coordinates
    sparse_row = [0 for product in unique_products] # Just one row—so all values 0
    product_col = {} # Take coordinates from previous product vectorization
    for product in unique_products:
        product_col[product] = tfidf_model.vocabulary_[product]

    # Make a dictionary to look up established idf weights by their terms
    idf_weights = dict(zip(tfidf_model.get_feature_names(),tfidf_model.idf_))

    # Get data from above dict pased on user selected products
    product_weight = {}
    for product in selected_products:
        if product in product_weight:
            product_weight[product] += idf_weights[product]
        else:
            product_weight[product] = idf_weights[product]

    # Get eveything in the same order
    sparse_col = []
    sparse_data = []
    for product in unique_products:
        sparse_col.append(product_col[product])
        sparse_data.append(product_weight[product])


    vector_mag = np.linalg.norm(sparse_data) # Normalize data
    normed_sparse_data = [weight / vector_mag for weight in sparse_data]


    # Construct matrix
    sparse_mtrx = csr_matrix((normed_sparse_data, (sparse_row, sparse_col)), shape=(1, len(tfidf_model.get_feature_names())))
    return sparse_mtrx

# Function that takes in products a user likes and outputs recommended products
def get_recommended_products(user_selection_sentence, k, n=10, include_prev=False, tfidf_model=tfidf, debug=False):

    product_mtrx = make_sparse(user_selection_sentence, tfidf_model=tfidf)
    cosine_sim = linear_kernel(product_mtrx, tfidf_matrix)[0]

    # Get the pairwsie similarity scores of all users
    sim_scores = list(enumerate(cosine_sim))

    # Sort the users based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top k scores
    if debug:
        sim_scores = sim_scores[1:(k+1)] # In debug mode I'll want to skip the best match since it will just be the input user
    else:
        sim_scores = sim_scores[:k]

    # Get the user indices
    user_indices = [i[0] for i in sim_scores]

    # The top k most similar users
    top_k_users = user_products_sentence__prior['user_id'].iloc[user_indices]

    # The products purchased from these top k users
    top_k_user_products = user_products_lookup.loc[top_k_users]
    top_k_user_products['product_id'] = top_k_user_products['product_id'].apply(literal_eval)
    product_list = [i for j in top_k_user_products['product_id'].tolist() for i in j]

    # Count the frequency of the items purchased and sort by popularity
    product_counts = dict(Counter(product_list))
    popular_products = sorted(product_counts, key=product_counts.get, reverse=True)

    top_items = [24852, 13176, 21137, 21903, 47209, 47766, 47626, 16797, 26209, 27845, 27966, 22935, 24964, 45007, 39275, 49683, 28204, 5876, 8277, 40706, 4920, 30391, 45066, 42265, 49235, 44632, 19057, 4605, 37646, 21616, 17794, 27104, 30489, 31717, 27086, 44359, 28985, 46979, 8518, 41950, 26604, 5077, 34126, 22035, 39877, 35951, 43352, 10749, 19660, 9076, 21938, 43961, 24184, 34969, 46667, 48679, 25890, 31506, 12341, 39928, 24838, 5450, 22825, 5785, 35221, 28842, 33731, 27521, 44142, 33198, 8174, 20114, 8424, 27344, 11520, 29487, 18465, 28199, 15290, 46906, 9839, 27156, 3957, 43122, 23909, 34358, 4799, 9387, 16759, 196, 42736, 38689, 4210, 41787, 41220, 47144, 7781, 33000, 20995, 21709, 19678, 40604, 30233, 34243, 37687, 24489, 42828, 5479, 432, 6184, 16185, 42768, 17948, 33754, 19348, 8193, 26369, 42585, 14992, 14947, 22963, 1463, 28849, 8021, 25659, 21405, 46676, 31343, 41844, 38293, 42701, 43789, 36011, 5025, 39475, 43295, 11777, 20842, 32689, 32655, 2295, 46802, 13870, 25146, 18531, 5212, 31553, 39408, 260, 36695, 10246, 24830, 38383, 43768, 1940, 11182, 18523, 18362, 21288, 6046, 44683, 29987, 890, 38777, 43772, 23734, 7948, 30450, 38456, 46969, 44910, 47734, 38159, 26620, 47672, 4957, 26165, 30776, 44987, 35939, 14678, 16349, 28289, 29447, 44422, 40310, 16953, 23375, 33787, 19048, 2078, 13984, 41290, 23165, 32864, 17600, 39812, 8859, 48364, 33120, 31683, 26940, 28465, 35108, 26283, 37158, 37067, 18370, 45, 41665, 35547, 3952, 10957, 2086, 2966, 43154, 24799, 40571, 10132, 43713, 40396, 18027, 45535, 1158, 25931, 7021, 38739, 13629, 42450, 20119, 15937, 42342, 32478, 48745, 45633, 21019, 34050, 34448, 48205, 39619, 17461, 45603, 30169, 31040, 38400, 13646, 36865, 40174, 21267, 17872, 5818, 26790, 39984, 14084, 23288, 21376, 32734, 33768, 37766, 46654, 12206, 329, 28934, 6187, 6348, 39993, 18441, 5646, 25340, 43086, 6104, 42356, 48775, 651, 10673, 47977, 19508, 39180, 40709, 19677, 10017, 27336, 13535, 13829, 42244, 11782, 27695, 36070, 28156, 6873, 49520, 49383, 8571, 24561, 13249, 49075, 16083, 24024, 40545, 40199, 30720, 17795, 25588, 35921, 17122, 11422, 8670, 31915, 35561, 42445, 14233, 7963, 17706, 3599, 21295, 44449, 36550, 19019, 30827, 20082, 35042, 25466, 33401, 1511, 42625, 5194, 8048, 23537, 2825, 16965, 38928, 21195, 19706, 18288, 13380, 15872, 7751, 4562, 9339, 22395, 38164, 46720, 40723, 15902, 35140, 4472, 3376, 13166, 29662, 2855, 28993, 7175, 13575, 13740, 12916, 40516, 45200, 7503, 5782, 19691, 5134, 40332, 12456, 21174, 5258, 10644, 39561, 4421, 33129, 44765, 20955, 46584, 22124, 44570, 3464, 43662, 13198, 10070, 32691, 7559, 37710, 36735, 35898, 37029, 34262, 1025, 18918, 42719, 11408, 18234, 44560, 23029, 45123, 3896, 9092, 28745, 18811, 26384, 35503, 35628, 47141, 28476, 25138, 21386, 47042, 46820, 21292, 11140, 21573, 38768, 17949, 1999, 14197, 46616, 32433, 47759, 37220, 39947, 24954, 41065, 21333, 25824, 48628, 43504, 1194, 44628, 5456, 38273, 15613, 35535, 15424, 27548, 24841, 39190, 27730, 42110, 29307, 14161, 49175, 14462, 31066, 16696, 26800, 9825, 5322, 27744, 8309, 11005, 41588, 11068, 27796, 27325, 2228, 7969, 22474, 9020, 16521, 5769, 3298, 1244, 13517, 27323, 2452, 28601, 2314, 21927, 32177, 13733, 12980, 15261, 19173, 21872, 20345, 41658, 18479, 49191, 20734, 22849, 4086, 36724, 17316, 5373, 26497, 20738, 44799, 20574, 5161, 35914, 22959, 37524, 6948, 12845, 4656, 17758, 47630, 26348, 25513, 33055, 34584, 11712]

    popular_products = [product for product in popular_products if product not in top_items]

    if include_prev:
        # Return the top n purchaed product by the k most similar users
        return popular_products[:n]

    else:
        # In this case we filter out items the user has already purchased
        prev_purchases = user_selection_sentence.split()
        prev_purchases = list(map(int, prev_purchases))
        popular_new_products = [product for product in popular_products if product not in prev_purchases]

        return popular_new_products[:n]

# Recommener function will output product_ids. Use this dict to then get human names for food
def product_id_to_name(product_ids):
    return [product_id_to_name_dict[id] for id in product_ids]

# print(product_id_to_name([24852]))

# Test case

# test_input = '196 14084 12427 26088 26405 196 10258 12427 13176 26088 13032 \
#                 196 12427 10258 25133 30450 196 12427 10258 25133 26405 196 12427 \
#                 10258 25133 10326 17122 41787 13176 196 12427 10258 25133 196 10258 12427 \
#                 25133 13032 12427 196 10258 25133 46149 49235 49235 46149 25133 196 10258 \
#                 12427 196 46149 39657 38928 25133 10258 35951 13032 12427'
#
# recommended_products = product_id_to_name(get_recommended_products(test_input, 200, include_prev=True, debug=True))
# print(recommended_products)
