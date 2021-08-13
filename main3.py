import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
	st.title("Welcome to my data science project")
	st.text("In this project I look into the trends of NYC taxis")

with dataset:
	st.header("NYC taxi dataset")

	taxi_data = pd.read_csv("data/taxi_data.csv")
	st.write(taxi_data.head())

	st.subheader("Pick-up location ID distribution on the NYC dataset")
	pulocation_dist = pd.DataFrame(taxi_data["start station id"].value_counts().head(50))

	st.bar_chart(pulocation_dist)

with features:
	st.header("The features I created")

	st.markdown("* **first feature:** I created this feature because of this ... I calculated it using this logic")
	st.markdown("* **second feature:** I created this feature because of this ... I calculated it using this logic")

with model_training:
	st.header("Time to train the model")
	st.text("Here you get to choose the parameters of the model")

	
##user input
	sel_col, disp_col = st.columns(2)

	max_depth = sel_col.slider("What should be the max depth of the model?", min_value = 10, max_value = 100)

	n_estimators = sel_col.selectbox("How many trees should there be?", options = [100,200,300,"No Limit"], index = 0)

	input_feature = sel_col.text_input("Which feature should be used as the input feature?", "start station id")

	regr = RandomForestRegressor(max_depth = max_depth, n_estimators=n_estimators)

	X = taxi_data[[input_feature]]
	y = taxi_data[["birth year"]]

	regr.fit(X,y)
	prediction = regr.predict(y)

	disp_col.subheader("Mean absolute error of the model is:")
	disp_col.write(mean_absolute_error(y,prediction))

	disp_col.subheader("Mean squared error of the model is:")
	disp_col.write(mean_squared_error(y,prediction))















