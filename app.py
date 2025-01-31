import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Used Car Sales Data Prediction Models", page_icon="🚗")

page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Prediction Models"])

df = pd.read_csv('data/df_cleaned.csv')
df_dummy = pd.read_csv('data/df_dummy.csv')


if page == "Home":
    st.title("Used Car Sales Data Prediction Models")
    st.subheader("An interacive app for exploring my analysis of the Used Car Sales Data!")
    st.write("""
        This app is designed to allow you to explore the Used Car Sales Data and my analysis on it in an interactive way. You can find explinations on all the various data points and view visualizations that further explore relationships between the different features. You can also see an explination of the data cleaning steps taken and a breakdown of the various models created!

            Use the sidebar to navigate through the sections!
            """)
    st.image('images/cars.jpg')

elif page == 'Data Overview':
    st.title("🔢 Data Overview")
    st.subheader("About the Data")
    st.write("""In this dataset we are presented with various features that describe the characteristics of many used cars.""")

    st.subheader("Data Summary")
    st.write("Select the options below to get some insights into the data!")
    if st.checkbox("Show Data Sample"):
        st.dataframe(df)

    if st.checkbox("Size of the Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Data Dictionary"):
        st.write("An explination of every feature in the dataset")
        st.image('images/dictionary.png')

    if st.checkbox("Average of each numerical feature"):
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        sel_col_num = st.selectbox("Select a numerical column from the Dataframe", num_cols)

        if sel_col_num:
            avg = round(df[sel_col_num].mean(),2)
            st.write(f"The average of {sel_col_num} is: {avg}!") 

    if st.checkbox("Breakdown of each Catagorical feature"):
        cat_cols = df.select_dtypes(exclude =['number']).columns.tolist()
        sel_col_cat = st.selectbox("Select a numerical column from the Dataframe", cat_cols)

        if sel_col_cat:
            st.write(f"The values for {sel_col_cat} are as follows:")
            st.write(df[sel_col_cat].value_counts())


elif page == "Exploratory Data Analysis": 
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.subheader("Select the type of Chart you would like to see:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Heatmaps'])
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
   

    if 'Histograms' in eda_type:
        num_col = df.select_dtypes(include=['number']).columns.tolist()
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols)
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Relationships compared to Price")
        b_selected_col = st.selectbox("Select a catagorical column for the box plot:", cat_cols)
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            st.plotly_chart(px.box(df, x=b_selected_col, y= 'price', title=chart_title))



    if 'Heatmaps' in eda_type:
        st.subheader("Heatmaps - Visualizing Correlations")
        if st.checkbox("Correlation of Features"):
            selected_col = st.selectbox("Select the variable you wish to see corelations for:", num_cols)
      
            if selected_col:
                correlation_matrix = df.corr(numeric_only=True)
                col_corr = correlation_matrix[[selected_col]].sort_values(by=selected_col, ascending=False)
            
        
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.heatmap(
                    col_corr,
                    vmin=-1,
                    vmax=1,
                    annot=True,
                    cmap='coolwarm',
                    ax=ax
                )
                st.pyplot(fig)
        
        if st.checkbox("Correlation to Target"):
            st.write("This heat map shows the correlation value for each feature comparred to the target variable of price. This heat map was created using dummies of the catagorical values.")             
              
            fig, ax = plt.subplots(figsize=(8, 25))
            sns.heatmap(df_dummy.corr(numeric_only = True)[['price']].sort_values(by = 'price', ascending = False),
            vmin=-1,
            vmax=1,
            annot=True,
            cmap='coolwarm',
            ax=ax
        )
            st.pyplot(fig)

elif page == "Prediction Models":
    st.title("Exploring the Models 📊")
    st.subheader("Model Explination and Scoring")
    st.write("Here I will breakdown each Model attempt, what was used, and how they scored.") 
    if st.checkbox("Model Scoring Explained"):
        st.write("Each model will be scored on two metrics:")
        if st.checkbox("RMSE Score"):
            st.write("The RMSE Score stands for Root Mean Squared Error Score. This will give us a postive value denoting the average amount off from the correct value we are. A lower number is always better. In this case an RMSE score of 20,000 means that our predictions are on average $20,000 off from the actual price.")
          
        if st.checkbox("R2 Score"):
            st.write("The R2 Score also known as the coefficient of determination is a statistical metric that measures how well a model fits data. Ranging from 0 to 1, the close the score is to a 1 the better.")

    if st.checkbox("Baseline Goal"):
        st.write("The Baseline Goal is the score of the model if it always predicts the average price. This Baseline gives us the score we need to beat to create a successful model.")   
        st.write("For this analysis we have a baseline RMSE score of 21247.24476")

    model_number = st.multiselect("Select which model you wish to view", ['Model #1', 'Model #2', 'Model #3', 'Model #4', "Overall Score Comparrison"])

    if 'Model #1' in model_number:
        st.subheader("Model #1")
        st.write("The first model was a Linear Regression Model using every available feature. This model served as the baseline to compare Linear Regression and Random Forest Regressor, used in model #2")
        st.subheader("The Scores")
        st.write("RMSE Score: 12640.83466")
        st.write("R2 Score: 0.6459")
        st.subheader("Conclusion")
        st.write("This model preformed significantly better than our Baseline model, this gives us a good starting place for tuning our models to be even better. I will be comparing this to the second model to determine which model type to refine for better results")

    if 'Model #2' in model_number:
        st.subheader("Model #2")
        st.write("The second model was a Random Forest Regressor model also using every available feature. I will determine which model to keep refining based on the first two scores.")
        st.subheader("The Scores")
        st.write("RMSE Score: 10714.27693")
        st.write("R2 Score: 0.7456")
        st.subheader("Conclusion")
        st.write("This second model preformed even better than the first. I will now continue with Random Forest Regressor and tune the paramaters to see if we can get a better result.")

    if 'Model #3' in model_number:
        st.subheader("Model #3")
        st.write("The third model used Random Forest Regressor with pramaters found by making a hypertuning script.")
        st.subheader("The Scores")
        st.write("RMSE Score: 10661.93078")
        st.write("R2 Score: 0.7481")
        st.subheader("Conclusion")
        st.write("Tuning the Random Forest Regressor parameters allowed us to get a marginally better score!")
        
    if 'Model #4' in model_number:
        st.subheader("Model #4")
        st.write("The last model was a test to see if reducing the number of features would help or hurt the model. I removed any features with a correlation coefficient lower than 0.05")
        st.subheader("The Scores")
        st.write("RMSE Score: 13773.9671")
        st.write("R2 Score: 0.5796")
        st.subheader("Conclusion")
        st.write("This final model did signifigantly worse, showing that removing features hurt the model instead of improving it. With that I would stick with the third model")

    if "Overall Score Comparrison" in model_number:
        st.subheader("Comparing the Scores of every model")
        st.image('images/scores.png')

        
