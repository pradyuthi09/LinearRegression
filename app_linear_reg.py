import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

#pip intsall streamlit seaborn scikit-learn matplotlib numpy pandas

#page configuration#
st.set_page_config("Linear Regression ",layout="centered")  # streamlit run app_linear_reg.py


#load css

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)    

load_css("style.css")

#Title#
st.markdown("""
        <div class="title">
        <h1>Linear Regression Model</h1>
        <p>predict <b>Tip Amount</b> from <b> Total Amount</b> using Linear Regression...</p>
           
        </div> """,unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------
#Load Data
@st.cache_data
def load_data():
    data=sns.load_dataset("tips")
    return data

df=load_data()

#Dataset Preview 

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

#----------------------------------------------------------------------------------------------------------------------

#prepare data 

x,y=df[['total_bill']],df['tip']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

# Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Metrics 

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)   
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)  
adjusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)


#Visualization

st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Total bill vs Tip Amount")
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df['tip'],color='blue',label='Actual Data',alpha=0.6)
ax.plot(
    np.sort(df['total_bill']),
    model.predict(scalar.transform(np.sort(df[['total_bill']], axis=0))),
    color='red',
    label='Regression Line'
)
ax.legend()
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.set_title("Total Bill vs Tip Amount")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

#------------------------------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
c1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# 🔽 reduce vertical gap here
st.markdown("<div style='margin-top:-20px'></div>", unsafe_allow_html=True)

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adjusted R² Score", f"{adjusted_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)


#m&c

st.markdown(f"""
<div class="card"
<h3>Model Intercept &co-eccicient</h3>
<p> <b> co-effiecient:</b> {model.coef_[0]:.3f}<br>
<b> Intercept :</b> {model.intercept_:.3f}</p>
</div>          """ ,unsafe_allow_html=True) 

#Prediction Box
# Prediction Box
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill Amount ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scalar.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
