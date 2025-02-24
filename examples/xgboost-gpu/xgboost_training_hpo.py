import numpy as np
import runhouse as rh
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ## First we encapsulate XGB HPO and training in a class
# We will send this training class to a remote instance with a GPU with Runhouse
class Trainer:
    def __init__(self):
        self.model = None
        self.dtrain = None
        self.dtest = None
        self.dval = None
        self.best_params = None

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

    def objective(self, trial):
        param = {
            "objective": "multi:softmax",
            "num_class": 10,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "eval_metric": ["mlogloss", "merror"],
            # Hyperparameters to optimize
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "seed": 42,
            "n_jobs": -1,
        }

        # Train with early stopping
        evals = [(self.dtrain, "train"), (self.dval, "val")]
        model = xgb.train(
            param,
            self.dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        # Return validation error (to be minimized)
        preds = model.predict(self.dval)
        return 1.0 - accuracy_score(self.dval.get_label(), preds)

    def run_optimization(self, n_trials=100):
        import optuna

        if not all([self.dtrain, self.dval]):
            raise ValueError("Data not loaded. Call load_data() first.")

        study = optuna.create_study(
            direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

        study.optimize(self.objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_params.update(
            {
                "objective": "multi:softmax",
                "num_class": 10,
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "eval_metric": ["mlogloss", "merror"],
                "seed": 42,
                "n_jobs": -1,
            }
        )

        print("\nBest trial:")
        print(f"Value (Error rate): {study.best_trial.value:.4f}")
        print("Params:")
        for key, value in study.best_trial.params.items():
            print(f"{key}: {value}")

        return self.best_params

    def train_with_best_params(self, num_rounds=1000):
        if not self.best_params:
            raise ValueError("No best parameters found. Run optimization first.")

        evals = [(self.dtrain, "train"), (self.dval, "val")]
        self.model = xgb.train(
            self.best_params,
            self.dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=True,
        )

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


# ## Set up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
if __name__ == "__main__":
    img = rh.Image().install_packages(
        ["xgboost", "pandas", "scikit-learn", "tensorflow", "numpy", "optuna"]
    )
    cluster = rh.compute(
        name="xgboost-gpu-cluster",
        instance_type="L4:1",
        provider="aws",
        image=img,
    ).up_if_not()

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

    # Now we send the training class to the remote cluster and invoke the training
    remote_trainer = rh.cls(Trainer).to(cluster, name="xgboost-gpu-training")
    remote_trainer.load_data()
    remote_trainer.run_optimization(n_trials=50)
    remote_trainer.train_with_best_params()
    remote_trainer.test_model()
    remote_trainer.save_model("fashion_mnist.model")

    # Finally, we tear down the cluster
    # cluster.teardown()
