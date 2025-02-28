import ray
from lightgbm_ray import predict, RayDMatrix, RayParams, train

from sklearn.metrics import mean_absolute_error, mean_squared_error


def date_preprocess(date_column, dataset):
    dataset = dataset.map_batches(
        lambda df: df.assign(
            day=df[date_column].dt.tz_convert(None).dt.day,
            month=df[date_column].dt.tz_convert(None).dt.month,
            dayofweek=df[date_column].dt.tz_convert(None).dt.dayofweek,
        ),
        batch_format="pandas",
    )

    return dataset, ["day", "month", "dayofweek"]


def load_data(data_path, sample_size=None):
    dataset = ray.data.read_parquet(data_path)
    if sample_size:
        dataset = dataset.random_sample(sample_size / dataset.count())

    return dataset


def write_data(dataset, write_data_path, sample_size=None):
    if sample_size:
        dataset = dataset.random_sample(sample_size / dataset.count())

    dataset.write_parquet(write_data_path)
    print(f"Saved dataset to {write_data_path}")


def save_model(model, path):
    import pickle

    print(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def preprocess_data(
    data_path,
    write_data_path,
    date_column=None,
    columns_to_retain=None,
    sample_size=None,
):
    dataset = load_data(data_path, sample_size)

    if date_column is not None:
        dataset, date_columns = date_preprocess(date_column, dataset)
    print("Processed dataset with schema:", dataset.schema())

    if columns_to_retain is not None:
        dataset = dataset.select_columns(columns_to_retain + date_columns)
    else:
        columns_to_retain = dataset.column_names()
    print("Columns to retain:", columns_to_retain)

    print("Writing preprocessed data to", write_data_path)
    write_data(dataset, write_data_path)
    return columns_to_retain + date_columns


def train_model(data_path, target_var, test_size=0.3, n_jobs=1):
    dataset = load_data(data_path)

    train_dataset, test_dataset = dataset.train_test_split(test_size)
    train_set = RayDMatrix(train_dataset, label="size")

    print("Training LightGBM model with Ray")

    evals_result = {}
    model = train(
        {
            "objective": "regression",
            "eval_metric": ["rmse"],
        },
        train_set,
        evals_result=evals_result,
        evals=[(train_set, "train")],
        verbose_eval=False,
        ray_params=RayParams(
            num_actors=n_jobs, cpus_per_actor=2  # Number of remote actors
        ),
    )

    print(evals_result["train"])

    save_model(model, "model.pkl")

    print("Evaluating model on test set")
    test_set = RayDMatrix(test_dataset.to_pandas(), label="size")
    y_test = test_dataset.select_columns([target_var])
    y_test = y_test.to_pandas().values.flatten()
    predictions = predict(model, test_set, ray_params=RayParams(num_actors=n_jobs))

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    return model
