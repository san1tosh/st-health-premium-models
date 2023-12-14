import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics


data = pd.read_csv("insurance.csv")
data["sex"].unique()
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"yes": 1, "no": 0})
data["region"] = data["region"].map(
    {"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4}
)
X = data.drop(["charges"], axis=1)
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gr = GradientBoostingRegressor()
gr.fit(X_train, y_train)
gr.fit(X, y)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.fit(X, y)
svr = SVR()
svr.fit(X_train, y_train)
svr.fit(X, y)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf.fit(X, y)


def main():
    html_temp = """
    <div style="background-color:lightblue;padding:16px>
    <h2 style="color:black;text-align:center>Health Insurance Cost Prediction Using ML</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # load the model
    model1 = gr
    model2 = lr
    model3 = svr
    model4 = rf

    p1 = st.slider("Enter Your Age", 18, 100)

    s1 = st.selectbox("Sex", ("Male", "Female"))
    if s1 == "Male":
        p2 = 1
    else:
        p2 = 0

    p3 = st.number_input("Enter Your BMI Value")
    p4 = st.slider("Enter Number of Children", 0, 10)

    s2 = st.selectbox("Smoker", ("Yes", "No"))
    if s2 == "Yes":
        p5 = 1
    else:
        p5 = 0

    p6 = st.selectbox(
        "Enter Your Region", ("Southwest", "Southeast", "Northwest", "Northeast")
    )
    if p6 == "Southwest":
        p6 = 1
    elif p6 == "Southeast":
        p6 = 2
    elif p6 == "Northwest":
        p6 = 3
    else:
        p6 = 4

    if st.button("Predict"):
        y_pred1 = model1.predict(X_test)
        y_pred2 = model2.predict(X_test)
        y_pred3 = model3.predict(X_test)
        y_pred4 = model4.predict(X_test)

        s1 = metrics.mean_absolute_error(y_test, y_pred1)
        s2 = metrics.mean_absolute_error(y_test, y_pred2)
        s3 = metrics.mean_absolute_error(y_test, y_pred3)
        s4 = metrics.mean_absolute_error(y_test, y_pred4)

        S = [s1, s2, s3, s4]

        if min(S) == s1:
            st.write("GRADIENT BOOSTING REGRESSOR IS BETTER")
            prediction = model1.predict([[p1, p2, p3, p4, p5, p6]])
            st.balloons()
            df1 = pd.DataFrame({"Actual": y_test[0:51], "gr": y_pred1[0:51]})
            print(df1)

            st.line_chart(
                df1,
                x=None,
                y=["Actual", "gr"],
                color=["#FF0000", "#0000FF"],
            )
            st.area_chart(
                df1,
                x=None,
                y=["Actual", "gr"],
                color=["#FF0000", "#0000FF"],
            )
            st.dataframe(df1)
            print(prediction)
            st.success("Insurance Amount is {} ".format(round(prediction[0], 2)))
            s1 = metrics.mean_absolute_error(y_test, y_pred1)
            st.write(f"Mean absolute error is {s1}")
            score1 = metrics.r2_score(y_test, y_pred1)
            st.write(f"R SQUARED (R2) metric is {score1}")

        elif min(S) == s2:
            st.write("LINEAR REGRESSION IS BETTER")
            prediction = model2.predict([[p1, p2, p3, p4, p5, p6]])
            st.balloons()
            df1 = pd.DataFrame({"Actual": y_test[0:51], "lr": y_pred2[0:51]})
            print(df1)

            st.line_chart(
                df1,
                x=None,
                y=["Actual", "lr"],
                color=["#FF0000", "#0000FF"],
            )
            st.area_chart(
                df1,
                x=None,
                y=["Actual", "lr"],
                color=["#FF0000", "#0000FF"],
            )
            st.dataframe(df1)
            print(prediction)
            st.success("Insurance Amount is {} ".format(round(prediction[0], 2)))
            st.write(f"Mean absolute error is {s2}")
            score2 = metrics.r2_score(y_test, y_pred2)
            st.write(f"R SQUARED (R2) metric is {score2}")

        elif min(S) == s3:
            st.write("SUPPORT VECTOR REGRESSION IS BETTER")
            prediction = model3.predict([[p1, p2, p3, p4, p5, p6]])
            st.balloons()
            df1 = pd.DataFrame({"Actual": y_test[0:51], "svr": y_pred3[0:51]})
            print(df1)

            st.line_chart(
                df1,
                x=None,
                y=["Actual", "svr"],
                color=["#FF0000", "#0000FF"],
            )
            st.area_chart(
                df1,
                x=None,
                y=["Actual", "svr"],
                color=["#FF0000", "#0000FF"],
            )
            st.dataframe(df1)
            print(prediction)
            st.success("Insurance Amount is {} ".format(round(prediction[0], 2)))
            st.write(f"Mean absolute error is {s3}")
            score3 = metrics.r2_score(y_test, y_pred3)
            st.write(f"R SQUARED (R2) metric is {score3}")

        elif min(S) == s4:
            st.write("RANDOM FOREST IS BETTER")
            prediction = model4.predict([[p1, p2, p3, p4, p5, p6]])
            st.balloons()
            df1 = pd.DataFrame({"Actual": y_test[0:51], "rf": y_pred4[0:51]})
            print(df1)

            st.line_chart(
                df1,
                x=None,
                y=["Actual", "rf"],
                color=["#FF0000", "#0000FF"],
            )
            st.area_chart(
                df1,
                x=None,
                y=["Actual", "rf"],
                color=["#FF0000", "#0000FF"],
            )
            st.dataframe(df1)
            print(prediction)
            st.success("Insurance Amount is {} ".format(round(prediction[0], 2)))
            st.write(f"Mean absolute error is {s4}")
            score4 = metrics.r2_score(y_test, y_pred4)
            st.write(f"R SQUARED (R2) metric is {score4}")

        else:
            pass


if __name__ == "__main__":
    main()
