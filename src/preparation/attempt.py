import pandas as pd
import numpy as np
import json
from scipy.sparse.linalg import svds

from src.lib.maths_functions import smooth_client_interest
from src.modeling.models import PopularityRecommender, CFRecommender



# the input csv must contain the following columns:
# 'Product ID', 'Customer ID', 'Product Name', 'Customer Name', 'Quantity'
csv_path = "./data/raw/superstore2011_2015.csv"
initial_df = pd.read_csv(csv_path, encoding="ISO-8859-1")


# For later. Create a .json to link Products and Customers IDs to their name
product_dict = dict(zip(initial_df['Product ID'], initial_df['Product Name']))
customer_dict = dict(zip(initial_df['Customer ID'], initial_df['Customer Name']))
with open(r'./data/processed/product_dict.json', 'w') as file:
    json.dump(product_dict, file)
with open(r'./data/processed/customer_dict.json', 'w') as file:
    json.dump(customer_dict, file)

# Create a new df grouped by Customer and Product. Smoothed quantity will be used to assess strength
grouped_df = initial_df.groupby(['Customer ID', 'Product ID'])['Quantity'].\
    sum().apply(smooth_client_interest).reset_index()
grouped_df.rename(columns={'Quantity': 'strength'}, inplace=True)
grouped_df.to_csv('data/processed/product_strength_by_customer.csv')


# Check which are the 5 most popular products and store them in a .csv
product_popularity_df = grouped_df.groupby('Product ID')['strength'].sum().\
    sort_values(ascending=False).reset_index()
product_popularity_df['Product Name'] = product_popularity_df['Product ID'].map(product_dict)
product_popularity_df.to_csv('data/processed/product_by popularity.csv')


# Collaborative filtering with Singular Value Decomposition (SVD)
customers_products_pivot_matrix_df = grouped_df.pivot(
    index='Customer ID', columns='Product ID', values='strength').fillna(0)
customers_products_pivot_matrix = customers_products_pivot_matrix_df.values

customers_ids = list(customers_products_pivot_matrix_df.index)

# The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
# Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(customers_products_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)
all_customers_predicted_purchases = np.dot(np.dot(U, sigma), Vt)

# convert the reconstructed matrix back to a pandas df
cf_preds_df = pd.DataFrame(all_customers_predicted_purchases,
                           columns=customers_products_pivot_matrix_df.columns,
                           index=customers_ids).transpose()




# call the models
# 1. Popularity
popularity_model = PopularityRecommender(product_popularity_df, initial_df)
customer = input('\n' + "Entrez le ID du client: ")
customer_name = customer_dict[customer]
print(customer_name)
print('\n' + "Parmi les items les plus populaires, voici ceux que %s n'a pas acheté: " % customer_name )
print("")
print(popularity_model.recommend_items(customer))

#2 Collaborative Filtering
cf_recommender_model = CFRecommender(cf_preds_df, initial_df)
print('\n' + "Voici 10 produits jamais achetés que ceux qui ressemblent à %s achètent aussi:" % customer_name)
print("")
print(cf_recommender_model.recommend_items(customer))


