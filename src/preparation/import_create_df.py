import pandas as pd
import json
from src.lib.maths_functions import smooth_client_interest
from src.modeling.models import PopularityRecommender


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


popularity_model = PopularityRecommender(product_popularity_df, initial_df)

customer = input("Entrez le ID d'un client: ")
print('\n' + "Voici parmi les items les plus populaires ceux que %s n'a pas acheté" %customer)
print(popularity_model.recommend_items(customer))

