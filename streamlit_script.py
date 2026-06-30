import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import plotly.graph_objects as go

from ACI_algo_v1 import infoACI_precompute_q, infoACI_precompute_q_optim

st.title("Online conformal testing with OnlineSCI")

st.markdown("""

$q_0$: $~$ Initialisation point $~~~~~~$ $(c, \\beta)$: $~$ Step-size $~$ $\\gamma_t = c \cdot t^{-\\beta}$

""")

# ============================
p = 2 

proby = np.array((.8, .2))
mu1 = 0
mu2 = 3

def generate_dataset(n, p):
    Y = np.random.choice(np.array((0, 1)), p=proby, size=n)
    X = []
    for i in range(n):
        if Y[i] == 0:
            X.append(np.random.normal(mu1, 1, size=p))
        elif Y[i] == 1:
            X.append(np.random.normal(mu2, 1, size=p))
    X = np.array(X)
    return X, Y

def f(x):
    prob_x = multivariate_normal(mu1*np.ones(p), np.ones(p)).pdf(x)*proby[0] 
    prob_x += multivariate_normal(mu2*np.ones(p), np.ones(p)).pdf(x)*proby[1]
    
    e1 = multivariate_normal(mu1*np.ones(p), np.ones(p)).pdf(x) / prob_x * proby[0]
    e2 = multivariate_normal(mu2*np.ones(p), np.ones(p)).pdf(x) / prob_x * proby[1]
    return np.array((e1, e2)).T 

def Score_func(y_pred, y, x):
     if len(y_pred.shape) > 1:
        return y_pred[:, 1]
     else: 
        return y_pred[1]

def info_fun(y_pred, y, x, s, q):
    return (s > q)*1

def Error_func(y_pred, y, x, s, q):
    return (y == 0).astype(float)

np.random.seed(1)
n = 20000
X, Y = generate_dataset(n, p)

Xt = X[:].copy()
Yt = Y[:len(Xt)].copy()

dates = np.arange(len(Xt))
T = len(Xt)

Ypredt = np.argmax(f(Xt), axis=1)
Scoret = Score_func(f(Xt), Yt, Xt)

# Computation of Pi(q)
alpha=.1

ni = 100000
Xi, Yi = generate_dataset(ni, p)

Ypredi = np.argmax(f(Xi), axis=1)*0 + 1
Scorei = Score_func(f(Xi), Yi, Xi)

PIq = []
for q in np.linspace(0, .99, 1000):
    info = info_fun(Ypredi, Yi, Xi, Scorei, q)
    err = Error_func(Ypredi, Yi, Xi, Scorei, q)
    ind = np.where(info==1)
    PIq.append(np.mean(err[ind]))
PIq = np.array(PIq)
q_star = np.linspace(0, 1, 1000)[np.argmin(np.abs(PIq-alpha))]

# Create a DataFrame from the generated data
c_names = ['date']
for i in range(p):
    c_names.append('value_' + str(i))

dt_Xt = np.vstack((dates.T, Xt.T)).T
dt_Xt = pd.DataFrame(dt_Xt, columns=c_names)
# Set the 'date' column as the index
dt_Xt.set_index('date', inplace=True)

# Create a DataFrame from the generated data
dt_Yt = pd.DataFrame({'date': dates, 'value': Yt})
# Set the 'date' column as the index
dt_Yt.set_index('date', inplace=True)

# Create a DataFrame from the generated data
dt_Ypredt = pd.DataFrame({'date': dates, 'value': Ypredt})
# Set the 'date' column as the index
dt_Ypredt.set_index('date', inplace=True)

# Create a DataFrame from the ge5nerated data
dt_Scoret = pd.DataFrame({'date': dates, 'value': Scoret})
# Set the 'date' column as the index
dt_Scoret.set_index('date', inplace=True)

t_range = np.arange(1, T+1)
BEST = infoACI_precompute_q(dt_Xt, dt_Yt, dt_Ypredt, dt_Scoret, info_fun, Error_func, t_range*0, alpha, q0=q_star, B=1)
BEST_ind_select = np.where(BEST[2]['value']==1)[0]
BEST_dtf = pd.DataFrame(np.cumsum(BEST[1]['value'][BEST_ind_select])/np.arange(1, len(BEST_ind_select)+1))
# ============================

q0 = st.slider("q0", 0., 1., 0.1)
c = st.slider("c", 0.1, 1., 0.5)
beta = st.slider("beta", 0.5, 0.9, 0.8)
alpha = .1

t_range = np.arange(1, T+1)
gamma = 1/(t_range**(beta))*c
dt_Qt_infoACIq, dt_errt_infoACIq, dtof_infot_infoACIq = infoACI_precompute_q_optim(dt_Xt, dt_Yt, dt_Ypredt, dt_Scoret, info_fun, Error_func, gamma, alpha, q0=q0, B=1)

ind_select = np.where(dtof_infot_infoACIq['value']==1)[0]
dtf = pd.DataFrame(np.cumsum(dt_errt_infoACIq['value'][ind_select])/np.arange(1, len(ind_select)+1))

# Create a layout with two columns
col1, col2 = st.columns(2)

# Place fig1 in the first column
with col1:
    fig1 = go.Figure(
        go.Scattergl(
            x=dt_Qt_infoACIq.index,
            y=dt_Qt_infoACIq["value"],
            mode="lines",
        )
    )
    fig1.add_hline(
        y=q_star,  # Y-value where the line will be drawn
        line_color="black",  # Color of the line
        line_width=2,  # Width of the line
    )
    fig1.update_layout(
        xaxis_title=dict(text="Times", font=dict(size=20)),
    yaxis_title=dict(text="q_t", font=dict(size=20))
    )
    st.plotly_chart(fig1, use_container_width=True)

# Place fig2 in the second column
with col2:
    fig2 = go.Figure(
        go.Scattergl(
            x=dtf.index,
            y=dtf["value"],
            mode="lines",
            showlegend=False
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=BEST_dtf.index,
            y=BEST_dtf["value"],
            mode="lines",
            line=dict(color="black"),
            showlegend=False
        )
    )
    fig2.update_layout(
        xaxis_title=dict(text="Times", font=dict(size=20)),
        yaxis_title=dict(text="FCP", font=dict(size=20))
    )
    st.plotly_chart(fig2, use_container_width=True)
