# ## Dask + LightGBM Training
# This script contains the implementation of a class that trains a LightGBM model using Dask.

class LightGBMModelTrainer:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.X_eval = None
        self.y_train = None
        self.y_test = None
        self.y_eval = None
        self.client = None
        self.mae = None
        self.mse = None
        self.features = None

    def load_client(self, ip="localhost", port="8786"):
        from dask.distributed import Client

        self.client = Client(f"tcp://{ip}:{port}")

    def load_data(self, data_path):
        import dask.dataframe as dd

        self.dataset = dd.read_parquet(data_path)
        print(self.dataset.columns)

    def train_test_split(self, target_var, features=None):
        from dask_ml.model_selection import train_test_split

        if features is None:
            features = self.dataset.columns.difference([target_var])

        X = self.dataset[features]
        y = self.dataset[target_var]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3
        )
        self.X_eval, self.X_train, self.y_eval, self.y_train = train_test_split(
            self.X_test, self.y_test, test_size=0.5
        )
        self.features = features

    def preprocess(self, date_column):
        self.dataset["day"] = self.dataset[date_column].dt.day
        self.dataset["month"] = self.dataset[date_column].dt.month
        self.dataset["dayofweek"] = self.dataset[date_column].dt.dayofweek
        self.dataset["hour"] = self.dataset[date_column].dt.hour
        return ["day", "month", "dayofweek", "hour"]

    def train_model(self):
        import lightgbm as lgb

        self.model = lgb.DaskLGBMRegressor(client=self.client)
        self.model.fit(
            self.X_train.to_dask_array(),
            self.y_train.to_dask_array(),
            eval_set=[(self.X_eval.to_dask_array(), self.y_eval.to_dask_array())],
        )
        print("Model trained successfully.", self.model)

    def test_model(self):
        from dask_ml.metrics import mean_absolute_error, mean_squared_error

        if self.model is None:
            raise Exception("Model not trained yet. Please train the model first.")
        if self.X_test is None or self.y_test is None:
            raise Exception(
                "Test data not loaded yet. Please load the test data first."
            )

        y_pred = self.model.predict(self.X_test.to_dask_array(lengths=True))
        self.mse = mean_squared_error(self.y_test.to_dask_array(lengths=True), y_pred)
        print(f"Mean Squared Error: {self.mse}")

        self.mae = mean_absolute_error(self.y_test.to_dask_array(lengths=True), y_pred)
        print(f"Mean Absolute Error: {self.mae}")

    def return_model_details(self):
        return {"features": self.features, "mse": self.mse, "mae": self.mae}

    def save_model(self, path, upload_to_s3=False):
        import cloudpickle

        if upload_to_s3:
            # Upload to s3
            import s3fs

            with s3fs.S3FileSystem() as s3:
                with s3.open(path, "wb") as f:
                    cloudpickle.dump(self.model, f)
        else:
            # Save to disk
            with open(path, "wb") as f:
                cloudpickle.dump(self.model, f)

    def load_model(self, path):
        import cloudpickle

        if path.startswith("s3://"):
            import boto3

            # Remove the "s3://" prefix and extract bucket and key
            s3 = boto3.client("s3")
            bucket, key = path[5:].split("/", 1)

            # Download the object to a bytes buffer
            response = s3.get_object(Bucket=bucket, Key=key)

            # Load model using cloudpickle
            self.model = cloudpickle.load(response["Body"].read())
        else:
            # Assume it's a local path
            with open(path, "rb") as f:
                self.model = cloudpickle.load(f)

    def predict(self, X):
        import dask.array as da
        import dask.dataframe as dd
        import numpy as np

        if isinstance(X, dd.DataFrame):  # Check for Dask DataFrame
            result = self.model.predict(X.to_dask_array())
        elif isinstance(X, da.Array):  # Check for Dask Array
            result = self.model.predict(X)
        elif isinstance(X, (list, tuple)):  # Check for list or tuple
            X = da.from_array(np.array(X).reshape(1, -1), chunks=(1, 7))
            result = self.model.predict(X)
        else:
            raise ValueError("Unsupported data type for prediction.")

        print(result)
        print(result.compute())
        return float(result.compute()[0])
