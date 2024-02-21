import streamlit as st
import pandas as pd
from sklearn import datasets
import plotly.express as px

def app():
    st.title("Chart & Plot Page")

    st.write("This is the visualizations of the iris dataset in bar chart and scatter plot.")

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    Y = pd.Series(iris.target, name="class")
    df = pd.concat([X, Y], axis=1)
    df["class"] = df["class"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    line_fig = px.bar(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        color="class",
        title="Bar Chart: Sepal Length vs Sepal Width",
    )

    line_fig.update_layout(xaxis_title="Sepal Length (cm)", yaxis_title="Sepal Width (cm)")

    scatter_fig = px.scatter(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        color="class",
        title="Scatter Plot: Sepal Length vs Sepal Width",
    )

    scatter_fig.update_layout(xaxis_title="Sepal Length (cm)", yaxis_title="Sepal Width (cm)")

    st.plotly_chart(line_fig)
    st.plotly_chart(scatter_fig)


