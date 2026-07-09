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


def estimate_q0_from_scores(scores: np.ndarray, y: np.ndarray, alpha: float, grid: np.ndarray) -> float:
    """For each threshold q in `grid`, selects points with score > q and computes
    the resulting false-coverage proportion PI(q) = mean(y == 0 | selected).
    Returns the q in the grid whose PI(q) is closest to alpha."""
    pi_q = np.full_like(grid, np.nan)

    for i, q_value in enumerate(grid):
        selected = scores > q_value
        if np.any(selected):
            pi_q[i] = np.mean(y[selected] == 0)

    return float(grid[np.nanargmin(np.abs(pi_q - alpha))])


@st.cache_data
def estimate_q_star(alpha: float, p: int, seed: int = 123, n_mc: int = 100_000) -> float:
    """Oracle threshold: PI(q) estimated on a large, fresh Monte-Carlo sample
    from the true data-generating distribution."""
    rng = np.random.default_rng(seed)
    x, y = generate_dataset(n_mc, p, rng)
    scores = posterior_class_one(x)
    grid = np.linspace(0.0, 0.99, 1000)
    return estimate_q0_from_scores(scores, y, alpha, grid)


@st.cache_data
def estimate_q0(scores: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Data-driven pilot estimate of q0: same PI(q) procedure as estimate_q_star,
    but computed on the actual observed sample (scores, y) instead of a fresh
    Monte-Carlo draw."""
    grid = np.linspace(0.0, 0.99, 1000)
    return estimate_q0_from_scores(scores, y, alpha, grid)


# ---------------------------------------------------------------------------
# Hardcoded grid-search over (c, beta) for OnlineSCI's step-size schedule.
# q0 is not searched over a grid: it is estimated directly (see estimate_q0).
# ---------------------------------------------------------------------------
c_range0 = np.linspace(0.1, 1, 10)
beta_range0 = np.linspace(0.5, 0.9, 10)

@st.cache_data
def grid_search_best_params(
    scores: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    alpha: float,
    q0_estim: float,
    c_range0: list[float],
    beta_range0: list[float],
) -> tuple[float, float]:
    """Brute-force search over (c, beta), with q0 fixed to q0_estim: maximize
    selection power subject to the empirical FCP staying below 1.2 * alpha."""
    A = []
    param = []

    for c_ in c_range0:
        for beta_ in beta_range0:
            gamma_ = c_ / (t**beta_)
            _, err_, selected_ = online_sci(
                scores=scores, y=y, gamma=gamma_, alpha=alpha, q0=q0_estim
            )

            # index 0 has no err/selection defined (see online_sci), so drop it
            err_ = err_[1:]
            selected_ = selected_[1:]

            ind_select = np.where(selected_)[0]
            fcp_ = np.sum(err_[ind_select]) / (len(ind_select) + 1)
            power_ = np.mean((1 - err_) * selected_)

            if fcp_ < alpha * 1.2:
                A.append(power_)
            else:
                A.append(-1)
            param.append((c_, beta_))

    c_best, beta_best = param[int(np.argmax(A))]
    return c_best, beta_best


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
    q0 = st.slider("q0", 0.0, 0.9, 0.3)
    c = st.slider("c", 0.0, 1.0, 0.5)
    beta = st.slider("beta", 0.5, 0.9, 0.8)
    n_burn = st.number_input(
        "Burn-in set size",
        min_value=100,
        max_value=20_000,
        value=500,
        step=100,
        help="Size of an independently drawn sample used only to estimate q0 (not part of the online run).",
    )

x, y, scores = make_data(n=n, p=p, seed=1)
q_star = estimate_q_star(alpha=alpha, p=p)

# Burn-in set: an independent draw (different seed, disjoint from the online
# sample) used only to estimate q0. The online phase runs on the full (x, y,
# scores) sample above.
x_burn, y_burn, scores_burn = make_data(n=n_burn, p=p, seed=2)

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

# q0, c, and beta are all estimated/selected on the independent burn-in set;
# the online phase then simply runs OnlineSCI with those fixed parameters.
q0_gs = estimate_q0(scores=scores_burn, y=y_burn, alpha=alpha)
t_burn = np.arange(1, n_burn + 1)
c_gs, beta_gs = grid_search_best_params(
    scores=scores_burn,
    y=y_burn,
    t=t_burn,
    alpha=alpha,
    q0_estim=q0_gs,
    c_range0=c_range0,
    beta_range0=beta_range0,
)
gamma_gs = c_gs / (t**beta_gs)
q_gs, err_gs, selected_gs = online_sci(
    scores=scores, y=y, gamma=gamma_gs, alpha=alpha, q0=q0_gs
)
fcp_gs = selected_running_mean(err_gs, selected_gs)
fcp_gs_x = np.where(selected_gs)[0]

st.caption(
    f"q0 estimated on {n_burn} independent burn-in samples: {q0_gs:.3f}. Grid "
    f"search best params on the same burn-in samples — c={c_gs}, beta={beta_gs}"
)

col1, col2 = st.columns(2, gap="large")
plot_height = 650

with col1:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scattergl(x=np.arange(n), y=q, mode="lines", name="q_t"))
    fig_q.add_trace(
        go.Scattergl(
            x=np.arange(n), 
            y=q_gs, 
            mode="lines", 
            name="q_t (automatic procedure)",
            line=dict(color="firebrick", width=2),
        )
    )
    fig_q.add_hline(y=q_star, line_color="black", line_width=2)
    fig_q.update_layout(height=plot_height, xaxis_title="Time", yaxis_title="q_t")
    st.plotly_chart(fig_q, use_container_width=True)

with col2:
    fig_fcp = go.Figure()
    fig_fcp.add_trace(go.Scattergl(x=fcp_x, y=fcp, mode="lines", name="OnlineSCI"))
    fig_fcp.add_trace(
        go.Scattergl(
            x=fcp_gs_x,
            y=fcp_gs,
            mode="lines",
            name="Automatic procedure",
            line=dict(color="firebrick", width=2),
        )
    )
    fig_fcp.add_trace(
        go.Scattergl(
            x=fcp_best_x,
            y=fcp_best,
            mode="lines",
            name="Oracle",
            line=dict(color="black", width=2),
        )
    )
    fig_fcp.update_layout(
        height=plot_height,
        xaxis_title="Time",
        yaxis_title="FCP",
    )
    st.plotly_chart(fig_fcp, use_container_width=True)
