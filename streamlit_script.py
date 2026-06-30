import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="OnlineSCI", layout="wide")


def generate_dataset(n: int, p: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    probs = np.array([0.8, 0.2])
    means = np.array([0.0, 3.0])

    y = rng.choice(2, size=n, p=probs)
    x = rng.normal(loc=means[y, None], scale=1.0, size=(n, p))
    return x, y


def posterior_class_one(x: np.ndarray) -> np.ndarray:
    """Returns P(Y = 1 | X = x) for the two-component Gaussian mixture."""
    log_prior_0 = np.log(0.8)
    log_prior_1 = np.log(0.2)

    log_phi_0 = -0.5 * np.sum(x**2, axis=1) + log_prior_0
    log_phi_1 = -0.5 * np.sum((x - 3.0) ** 2, axis=1) + log_prior_1

    m = np.maximum(log_phi_0, log_phi_1)
    return np.exp(log_phi_1 - m) / (np.exp(log_phi_0 - m) + np.exp(log_phi_1 - m))


def online_sci(
    scores: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    q0: float,
    bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the OnlineSCI threshold update."""
    n = len(y)

    q = np.empty(n)
    err = np.empty(n)
    selected = np.zeros(n, dtype=bool)

    q[0] = q0
    err[0] = np.nan

    n_selected = 0

    for t in range(1, n):
        q_prev = q[t - 1]
        err_t = float(y[t] == 0)

        if scores[t] > q_prev:
            selected[t] = True
            n_selected += 1
            q_new = q_prev + gamma[n_selected - 1] * (err_t - alpha)
        else:
            q_new = q_prev

        if bound > 0.0 and q_new > bound:
            q_new = q0

        q[t] = q_new
        err[t] = err_t

    return q, err, selected


def selected_running_mean(err: np.ndarray, selected: np.ndarray) -> np.ndarray:
    values = err[selected]
    if len(values) == 0:
        return np.array([])
    return np.cumsum(values) / np.arange(1, len(values) + 1)


@st.cache_data
def make_data(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, y = generate_dataset(n, p, rng)
    scores = posterior_class_one(x)
    return x, y, scores


@st.cache_data
def estimate_q_star(alpha: float, p: int, seed: int = 123, n_mc: int = 100_000) -> float:
    rng = np.random.default_rng(seed)
    x, y = generate_dataset(n_mc, p, rng)
    scores = posterior_class_one(x)

    grid = np.linspace(0.0, 0.99, 1000)
    pi_q = np.full_like(grid, np.nan)

    for i, q_value in enumerate(grid):
        selected = scores > q_value
        if np.any(selected):
            pi_q[i] = np.mean(y[selected] == 0)

    return float(grid[np.nanargmin(np.abs(pi_q - alpha))])


st.title("Online conformal testing with OnlineSCI")

st.markdown(
    r"""
$q_0$: initial threshold.  
$(c, \beta)$: step-size parameters with $\gamma_t = c t^{-\beta}$.
"""
)

p = 2
alpha = 0.1

with st.sidebar:
    n = st.number_input("Sample size", min_value=1_000, max_value=100_000, value=20_000, step=1_000)
    q0 = st.slider("q0", 0.0, 0.9, 0.1)
    c = st.slider("c", 0.0, 1.0, 0.5)
    beta = st.slider("beta", 0.5, 0.9, 0.8)

x, y, scores = make_data(n=n, p=p, seed=1)
q_star = estimate_q_star(alpha=alpha, p=p)

t = np.arange(1, n + 1)
gamma = c / (t**beta)

q, err, selected = online_sci(scores=scores, y=y, gamma=gamma, alpha=alpha, q0=q0)
fcp = selected_running_mean(err, selected)
fcp_x = np.where(selected)[0]

q_best, err_best, selected_best = online_sci(
    scores=scores,
    y=y,
    gamma=np.zeros(n),
    alpha=alpha,
    q0=q_star,
)
fcp_best = selected_running_mean(err_best, selected_best)
fcp_best_x = np.where(selected_best)[0]

col1, col2 = st.columns(2, gap="large")
plot_height = 650

with col1:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scattergl(x=np.arange(n), y=q, mode="lines", name="q_t"))
    fig_q.add_hline(y=q_star, line_color="black", line_width=2)
    fig_q.update_layout(height=plot_height, xaxis_title="Time", yaxis_title="q_t")
    st.plotly_chart(fig_q, use_container_width=True)

with col2:
    fig_fcp = go.Figure()
    fig_fcp.add_trace(go.Scattergl(x=fcp_x, y=fcp, mode="lines", name="OnlineSCI"))
    fig_fcp.add_trace(
        go.Scattergl(
            x=fcp_best_x,
            y=fcp_best,
            mode="lines",
            name="Oracle",
            line=dict(color="black", width=2),
        )
    )
    fig_fcp.update_layout(height=plot_height, xaxis_title="Time", yaxis_title="FCP")
    st.plotly_chart(fig_fcp, use_container_width=True) 
