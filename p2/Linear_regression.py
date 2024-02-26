import sklearn
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model

df = pd.read_csv("clean_data.csv")

df["IsHoliday"] = df["IsHoliday"].astype(int)
df["Super_Bowl"] = df["Super_Bowl"].astype(int)
df["Labour_Day"] = df["Labour_Day"].astype(int)
df["Thanksgiving"] = df["Thanksgiving"].astype(int)
df["Christmas"] = df["Christmas"].astype(int)

df = df[
    [
        "weekly_sales",
        "IsHoliday",
        "Super_Bowl",
        "Labour_Day",
        "Thanksgiving",
        "Christmas",
    ]
]

predict = "IsHoliday"

x = np.array(df.drop([predict], axis=1))
y = np.array(df[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1
)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc * 100, "%")


print("co: ", linear.coef_)
print("intercept: ", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(1, 100):
    print(round(prediction[x]), x_test[x], y_test[x])
