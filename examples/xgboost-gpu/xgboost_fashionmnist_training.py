import kubetorch as kt

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ## Encapsulate XGB training
# We define a Trainer class locally, but we will send this training class to remote compute with a GPU.
# This, like most XGB models, can be run on CPU, but works 100x faster on GPU.
class Trainer:
    def __init__(self):
        self.model = None
        self.dtrain = None
        self.dtest = None
        self.dval = None

    def load_data(self):
        import tensorflow as tf  # Imports in the function are only required on remote Image, not local env

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # Preprocess features
        X_train = self._preprocess_features(X_train)
        X_test = self._preprocess_features(X_test)

        # Split test into validation and test
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        self.dval = xgb.DMatrix(X_val, label=y_val)

    @staticmethod
    def _preprocess_features(X):
        X = X.reshape(X.shape[0], -1).astype(np.float32)
        return X / 255.0

    def train_model(self, params, num_rounds):
        if not all([self.dtrain, self.dval]):
            raise ValueError("Data not loaded. Call load_data() first.")

        evals = [(self.dtrain, "train"), (self.dval, "val")]
        self.model = xgb.train(
            params,
            self.dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=True,
        )

    def test_model(self):
        preds = self.model.predict(self.dtest)
        accuracy = accuracy_score(self.dtest.get_label(), preds)
        print(f"Test accuracy: {accuracy:.4f}")
        print(
            "\nClassification Report:\n",
            classification_report(self.dtest.get_label(), preds),
        )

        return preds, accuracy

    def predict(self, X):
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X)
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)


# ## Set up compute and run training
#
# Now, we define the main function that will run locally when we run this script and set up
# our desired remote compute.
if __name__ == "__main__":
    img = kt.images.ubuntu().pip_install(
        ["xgboost", "pandas", "scikit-learn", "tensorflow", "numpy"]
    )
    compute = kt.Compute(gpus="L4:1", image=img)

    train_params = {
        "objective": "multi:softmax",
        "num_class": 10,
        "eval_metric": ["mlogloss", "merror"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "gpu_hist",  # Using a GPU here reduces training time by ~99%
        "predictor": "gpu_predictor",
        "seed": 42,
        "n_jobs": -1,
    }

    # Now we send the training class to the remote compute and invoke the training
    remote_trainer = kt.cls(Trainer).to(compute, name="xgboost-gpu-training")
    remote_trainer.load_data()
    remote_trainer.train_model(train_params, num_rounds=100)
    remote_trainer.test_model()
    remote_trainer.save_model("fashion_mnist.model")
