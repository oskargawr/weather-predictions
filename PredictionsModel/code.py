# %%
import pandas as pd

weather = pd.read_csv("WeatherData/weather_jfk.csv", index_col="DATE")

# %%
weather

# %%
# weather = weather.loc['1985':]
weather


# %% [markdown]
# filtered for newer data than 1985, so that i get more non null columns

# %%
null_pct = weather.apply(pd.isnull).sum() / weather.shape[0]

# %%
null_pct

# %% [markdown]
# removing the data where the null percentage is more than 5% in order to clean the data

# %%
valid_columns = weather.columns[null_pct < 0.05]

# %%
valid_columns

# %%
weather = weather[valid_columns].copy()

# %%
weather.columns = weather.columns.str.lower()

# %%
weather

# %%
weather.apply(pd.isnull).sum()

# %%
weather.ffill(inplace=True)

# %% [markdown]
# replacing all missing values with a value from the previous row, so that the data is continuous

# %%
weather.apply(pd.isnull).sum()

# %%
weather.dtypes

# %%
weather.index

# %%
weather.index = pd.to_datetime(weather.index)

# %% [markdown]
# changing the index data type to data

# %%
weather.index.year.value_counts().sort_index()

# %%
weather["tmax"].plot()

# %%
weather["target"] = weather.shift(-1)["tmax"]
weather

# %%
weather = weather.ffill()
weather

# %%
numeric_columns = weather.select_dtypes(include=["float", "int"]).columns
weather[numeric_columns].corr()


# %%
from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.1)

# %%
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictors


# %%
def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)

        combined.columns = ["actual", "predicted"]

        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions = backtest(weather, rr, predictors)
predictions

# %%
from sklearn.metrics import mean_absolute_error

mean_absolute_error(predictions["actual"], predictions["predicted"])


# %%
def pct_diff(old, new):
    return (new - old) / old


def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"

    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])

    return weather


rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

weather

# %%
weather = weather.iloc[14:, :]
weather

# %%
weather = weather.fillna(0)


# %%
def expand_mean(df):
    return df.expanding(1).mean()


for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = (
        weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    )
    weather[f"day_avg_{col}"] = (
        weather[col].groupby(weather.index.day, group_keys=False).apply(expand_mean)
    )


weather

# %%
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictors

# %%
predictions = backtest(weather, rr, predictors)
predictions

mean_absolute_error(predictions["actual"], predictions["predicted"])


# %%
predictions.sort_values("diff", ascending=False)

# %%
weather.loc["1990-03-07":"1990-03-17"]

# %%
predictions["diff"].round().value_counts().sort_index().plot()

# %% [markdown]
# try xgboost, lightgbm, catboost, and random forest to see which one is the best

# %% [markdown]
# -- XGBOOST SECTION --

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    weather[predictors], weather["target"], test_size=0.2, random_state=42
)

# %%
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBRegressor

pipe = Pipeline([("target_encoder", TargetEncoder()), ("xgb", XGBRegressor())])
pipe.fit(X_train, y_train)


# %%
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, pipe.predict(X_test))


# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    "xgb__n_estimators": [100, 200, 300],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.1, 0.3],
}

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)

params = grid.best_params_


# %%
def backtest_xgboost(weather, model, predictors, model_params, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.set_params(**model_params)
        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)
        combined.columns = ["actual", "predicted"]
        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions_xgboost = backtest_xgboost(weather, pipe, predictors, params)
mean_absolute_error(predictions_xgboost["actual"], predictions_xgboost["predicted"])

# %%
predictions_xgboost

# %%
import joblib

joblib.dump(grid, "weather_model.pkl")

model = joblib.load("weather_model.pkl")

model.predict(X_test)

model.best_params_

model.best_score_

model.best_estimator_

model.best_estimator_.named_steps["xgb"].feature_importances_

mean_absolute_error(y_test, model.predict(X_test))

# %%
import matplotlib.pyplot as plt

plt.barh(X_train.columns, model.best_estimator_.named_steps["xgb"].feature_importances_)
plt.show()

# %%
predictions_xgboost["diff"].round().value_counts().sort_index().plot()

# %% [markdown]
# -- NEURAL NETWORKS --

# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_model(input_shape):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_shape,)),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mae")

    return model


model = build_model(X_train.shape[1])


# %%
def backtest_neural_network(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        X_train, y_train = train[predictors], train["target"]
        X_test, y_test = test[predictors], test["target"]

        model.fit(
            X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0
        )

        predictions = model.predict(
            X_test
        ).flatten()  # Otrzymujemy przewidywane wartości

        predictions_df = pd.DataFrame(
            {
                "actual": y_test.values,
                "predicted": predictions,
                "diff": (y_test.values - predictions),
            },
            index=y_test.index,
        )

        all_predictions.append(predictions_df)

    return pd.concat(all_predictions)


# %%
predictions_neural_network = backtest_neural_network(weather, model, predictors)
joblib.dump(predictions_neural_network, "predictions_neural_network.pkl")
mean_absolute_error(
    predictions_neural_network["actual"], predictions_neural_network["predicted"]
)

# %%
predictions_neural_network.sort_index()

# %%
from tensorflow.keras import regularizers

# dodanie regularyzacji L1 do warstwy Dense
model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],),
        kernel_regularizer=regularizers.l1(0.01),
    )
)

predictions_neural_network = backtest_neural_network(weather, model, predictors)
joblib.dump(predictions_neural_network, "predictions_neural_network_l1.pkl")
mean_absolute_error(
    predictions_neural_network["actual"], predictions_neural_network["predicted"]
)

# %%
model.add(
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],),
        kernel_regularizer=regularizers.l2(0.01),
    )
)

predictions_neural_network = backtest_neural_network(weather, model, predictors)
joblib.dump(predictions_neural_network, "predictions_neural_network_l2.pkl")
mean_absolute_error(
    predictions_neural_network["actual"], predictions_neural_network["predicted"]
)

# %%
mlp_l2 = joblib.load("predictions_neural_network_l2.pkl")
mlp_l2

# %%
mlp_l1 = joblib.load("predictions_neural_network_l1.pkl")
mlp_l1

# %% [markdown]
# -- random forest --

# %%
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, max_depth=5)


# %%
def backtest_random_forest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)
        combined.columns = ["actual", "predicted"]
        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions_rf = backtest_random_forest(weather, rf_model, predictors)
joblib.dump(predictions_rf, "predictions_rf.pkl")
mean_absolute_error(predictions_rf["actual"], predictions_rf["predicted"])

# %% [markdown]
# -- naive bayes --

# %%
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()


# %%
def backtest_naive_bayes(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)
        combined.columns = ["actual", "predicted"]
        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions_nb = backtest_naive_bayes(weather, nb_model, predictors)
joblib.dump(predictions_nb, "predictions_nb.pkl")
mean_absolute_error(predictions_nb["actual"], predictions_nb["predicted"])


# %%
predictions_nb

# %% [markdown]
# -- knn --

# %%
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)


# %%
def backtest_knn(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)
        combined.columns = ["actual", "predicted"]
        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions_knn = backtest_knn(weather, knn_model, predictors)
joblib.dump(predictions_knn, "predictions_knn.pkl")
mean_absolute_error(predictions_knn["actual"], predictions_knn["predicted"])

# %% [markdown]
# -- lasso regression --

# %%
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model2 = Lasso(alpha=0.01)
lasso_model3 = Lasso(alpha=0.2)
lasso_model4 = Lasso(alpha=0.001)


# %%
def backtest_lasso(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i, :]
        test = weather.iloc[i : (i + step), :]

        model.fit(train[predictors], train["target"])

        predictions = model.predict(test[predictors])

        predictions = pd.Series(predictions, index=test.index)
        combined = pd.concat([test["target"], predictions], axis=1)
        combined.columns = ["actual", "predicted"]
        combined["diff"] = (combined["actual"] - combined["predicted"]).abs()

        all_predictions.append(combined)

    return pd.concat(all_predictions)


# %%
predictions_lasso = backtest_lasso(weather, lasso_model, predictors)
joblib.dump(predictions_lasso, "predictions_lasso.pkl")
mean_absolute_error(predictions_lasso["actual"], predictions_lasso["predicted"])

# %%
# compare other lasso models to see which has the best performance

predictions_lasso2 = backtest_lasso(weather, lasso_model2, predictors)
joblib.dump(predictions_lasso2, "predictions_lasso2.pkl")
print(
    mean_absolute_error(predictions_lasso2["actual"], predictions_lasso2["predicted"])
)

predictions_lasso3 = backtest_lasso(weather, lasso_model3, predictors)
joblib.dump(predictions_lasso3, "predictions_lasso3.pkl")
print(
    mean_absolute_error(predictions_lasso3["actual"], predictions_lasso3["predicted"])
)


# %%
predictions_lasso4 = backtest_lasso(weather, lasso_model4, predictors)
joblib.dump(predictions_lasso4, "predictions_lasso4.pkl")
print(
    mean_absolute_error(predictions_lasso4["actual"], predictions_lasso4["predicted"])
)
