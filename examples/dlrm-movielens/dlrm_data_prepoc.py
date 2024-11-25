import runhouse as rh 


def preprocess_data(s3_base_path): 
    import ray
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    genome_scores = ray.data.read_csv(f'{s3_base_path}/genome-scores.csv')
    genome_tags = ray.data.read_csv(f'{s3_base_path}/genome-tags.csv')
    movies = ray.data.read_csv(f'{s3_base_path}/movies.csv')
    ratings = ray.data.read_csv(f'{s3_base_path}/ratings.csv')
    tags = ray.data.read_csv(f'{s3_base_path}/tags.csv')

    # Join ratings and movies datasets
    def join_ratings_movies(row):
        movie = movies.filter(lambda m: m['movieId'] == row['movieId']).take(1)[0]
        return {**row, **movie}

    data = ratings.map(join_ratings_movies)

    # Join genome scores with genome tags to get descriptive tags
    genome_combined = genome_scores.map(lambda row: {
        **row,
        'tag': genome_tags.filter(lambda tag: tag['tagId'] == row['tagId']).take(1)[0]['tag']
    })

    # Aggregate genome data to create features for each movie
    def aggregate_genomic_features(block_iter):
        df = pd.DataFrame(block_iter)
        genomic_features_df = df.pivot_table(
            index='movieId',
            columns='tag',
            values='relevance',
            fill_value=0
        )
        return genomic_features_df.reset_index()

    genomic_features = genome_combined.map_batches(aggregate_genomic_features).to_pandas()

    # Add genomic features to the movie dataset
    def add_genomic_features(row):
        movie_id = row['movieId']
        genomic_features_row = genomic_features.loc[genomic_features['movieId'] == movie_id]
        if not genomic_features_row.empty:
            genomic_data = genomic_features_row.iloc[0].to_dict()
            row.update(genomic_data)
        return row

    data = data.map(add_genomic_features)

    # Feature engineering
    def feature_engineering(row):
        # Extracting year from title
        title = row.get('title', '')
        year = int(title[-5:-1]) if '(' in title and title[-5:-1].isdigit() else 0
        row['year'] = year
        # Fill missing tag values
        row['tag'] = row['tag'] if pd.notnull(row['tag']) else ''
        return row

    data = data.map(feature_engineering)

    # Encode categorical data
    def encode_categorical(data):
        df = pd.DataFrame(data)
        user_encoder = LabelEncoder()
        movie_encoder = LabelEncoder()
        df['userId'] = user_encoder.fit_transform(df['userId'])
        df['movieId'] = movie_encoder.fit_transform(df['movieId'])
        return df.to_dict(orient="records")

    encoded_data = data.map_batches(encode_categorical)

    # Save the preprocessed data to a CSV file
    processed_data_df = encoded_data.to_pandas()
    processed_data_file = 'processed_movielens_data_with_genomic_features.csv'
    processed_data_df.to_csv(processed_data_file, index=False)

    print(f"Data preprocessing complete. Processed data saved to {processed_data_file}.")


if __name__ == "__main__":
    num_nodes = 2
    
    cluster = rh.cluster(
        name="rh-preprocessing",
        instance_type="CPU:4+",
        provider="aws",
        region="us-east-1",
        num_nodes = num_nodes,
        default_env=rh.env(
            reqs=[
                "sklearn",
                "pandas",
                "ray[data]",
            ],
        ),
    ).up_if_not()

    remote_preprocess = rh.function(preprocess_data).to(cluster).distribute('ray')
    remote_preprocess('s3://rh-demo-external/dlrm-training-example/raw_data/')