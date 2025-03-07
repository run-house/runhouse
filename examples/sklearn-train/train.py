import numpy as np
import runhouse as rh
import statsmodels.api as sm

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2
    )

    return X_train, X_test, y_train, y_test


def train_with_skl():
    X_train, X_test, y_train, y_test = load_data()

    # Standardize the data (important for many models)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def train_with_statsmodel():
    X_train, X_test, y_train, y_test = load_data()

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add intercept term
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Train a logistic regression model
    model = sm.MNLogit(y_train, X_train)
    result = model.fit()

    # Make predictions
    y_pred_prob = result.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print model summary
    print(result.summary())


if __name__ == "__main__":
    img = rh.Image(name="sklearn-training").install_packages(
        ["scikit-learn", "statsmodels", "numpy"]
    )

    compute = rh.compute(
        name="cpu-train", num_cpus="4+", image=img, provider="aws"
    ).up_if_not()

    remote_train_with_skl = rh.function(train_with_skl).to(compute)
    remote_train_with_statsmodel = rh.function(train_with_statsmodel).to(compute)

    remote_train_with_skl()
    remote_train_with_statsmodel()

    # compute.teardown()
