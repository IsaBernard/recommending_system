class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, customer_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['Product ID'].isin(items_to_ignore)] \
            .sort_values('strength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='Customer ID',
                                                          right_on='Customer ID')[
                ['strength', 'Customer ID']]

        return recommendations_df


class CFRecommender:

    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, customer_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_customer_predictions = self.cf_predictions_df[customer_id].sort_values(ascending=False) \
            .reset_index().rename(columns={customer_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_customer_predictions[~sorted_customer_predictions['Product ID'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"customers_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='Product ID',
                                                          right_on='Product ID')[['Product Id', 'Product Name']]

        return recommendations_df
