import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns", css_file="custom.css")

theme = marimo.ui.toggle(left_label="Light", right_label="Dark", value=True)

import plotly.io as pio
pio.templates["my_dark"] = pio.templates["plotly_dark"]
pio.templates["my_dark"].layout.font.color = "white"
pio.templates.default = "my_dark"

@app.cell
async def _():
    import micropip
    await micropip.install("pandas==2.2.3")

@app.cell(hide_code=True)
def intro(mo):
    mo.md(
        r"""
    ## Modelling with the exponential family of distributions

    In this notebook I will try to show how the different members of the exponential family relate to each other in the context of event/survival modelling. 

    #### Exponential distribution
    The exponential distribution is a continuous probability distribution often used to model the time until an event occurs, such as the lifespan of a product or the time between arrivals in a Poisson process. It is characterized by its rate parameter, λ (lambda), which determines how quickly events occur.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    slider = mo.ui.slider(start=0.01, stop=1, step=0.01, value=0.1, label="Choose your rate (lambda):")

    slider
    return mo, slider


@app.cell(hide_code=True)
def _(mo, slider):
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import math
    import pandas as pd


    def make_exponential_plot(lam, xmax=30, npts=400):
        # median (50/50) time: ln(2)/λ; handle lam=0 gracefully
        t_half = math.log(2)/lam if lam > 0 else float("inf")

        # ensure the domain shows the median if it's within reason
        if np.isfinite(t_half) and t_half > xmax:
            xmax = t_half * 1.05  # just extend a bit so the line is visible

        x = np.linspace(0, xmax, npts)
        y = np.exp(-lam * x)

        # base line
        fig = px.line(
            x=x, y=y,
            labels={'x': 'x', 'y': 'f(x)'},
            title=f"f(x) = exp(-{lam}·x)"
        )

        # make the line yellow
        fig.update_traces(line=dict(color='yellow'))

        # slightly opaque yellow area under the curve
        fig.add_scatter(
            x=x, y=y,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(255, 215, 0, 0.25)",  # light transparent yellow
            hoverinfo="skip",
            showlegend=False
        )

        # vertical line at the 50/50 point (median of the exponential distribution)
        if np.isfinite(t_half):
            fig.add_vline(
                x=t_half,
                line_dash="dash",
                line_color="red",
                annotation_text=f"50/50 at x = {t_half:.3f}",
                annotation_position="top right"
            )

        fig.update_yaxes(rangemode="tozero")

        return fig

    exp_fig = make_exponential_plot(slider.value)

    mo.ui.plotly(exp_fig)
    return go, make_exponential_plot, math, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Gamma distribution

    So the exponential distribution shows the probability of how long it takes for an event to occur. But when we want to for instance model inventory, and we want to know when to restock, we might want to know when it will have happened n times.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gamma_k_slider = mo.ui.slider(start=1, stop=100, step=1, value=1, label="Choose amount of events:")

    gamma_k_slider
    return (gamma_k_slider,)


@app.cell(hide_code=True)
def _(gamma_k_slider, go, math, mo, np, slider):
    def make_gamma_plot(k=3, lam=0.7, xmax=300, npts=600):
        """
        Create a Plotly figure for the Gamma(k, λ) density (time to k-th event),
        """
        if k <= 0 or lam <= 0:
            raise ValueError("Both k (shape) and lam (rate) must be > 0.")

        x = np.linspace(0, xmax, npts)
        # Gamma(k, λ) pdf: f(x) = λ * e^{-λx} * (λx)^{k-1} / Γ(k), for x >= 0
        gamma_k = math.gamma(k)
        y = lam * np.exp(-lam * x) * (lam * x) ** (k - 1) / gamma_k

        fig = go.Figure()

        # slightly opaque green area under the curve
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0, 128, 0, 0.25)",
            hoverinfo="skip",
            showlegend=False
        ))

        # green outline line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color="green", width=2),
            name="Gamma PDF"
        ))

        fig.update_layout(
            title=f"Gamma(k={k}, λ={lam}) density",
            xaxis_title="x (time)",
            yaxis_title="f(x)"
        )
        fig.update_yaxes(rangemode="tozero")

        return fig

    gamma_fig = make_gamma_plot(k=gamma_k_slider.value, lam=slider.value)

    mo.ui.plotly(gamma_fig)
    return (make_gamma_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Poisson distribution

    What also can be useful is to look at how the amount of events within a timeframe is distributed. We can show this with a poisson.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    poisson_days_slider = mo.ui.slider(start=1, stop=100, step=1, value=1, label="n days:")

    poisson_days_slider
    return (poisson_days_slider,)


@app.cell(hide_code=True)
def _(go, math, mo, np, poisson_days_slider, slider):
    def make_poisson_plot(lam=0.7, period=8.0, k_max=None):
        """
        Create a Plotly figure for the Poisson(μ) PMF representing the
        number of events within a period, where μ = lam * period.

        """
        if lam < 0 or period < 0:
            raise ValueError("lam and period must be non-negative.")
        mu = lam * period  # Poisson mean/variance

        # Choose support up to a few std devs above the mean
        if k_max is None:
            k_max = max(15, int(mu + 6 * math.sqrt(mu + 1e-12)))
        k = np.arange(0, k_max + 1)

        # Poisson PMF (stable log form)
        log_pmf = k * (math.log(mu) if mu > 0 else -np.inf) - mu - np.array([math.lgamma(i + 1) for i in k])
        pmf = np.exp(log_pmf)
        if mu == 0:
            pmf = np.zeros_like(k, dtype=float)
            pmf[0] = 1.0

        fig = go.Figure()

        # Bars for PMF — slightly transparent purple
        fig.add_trace(go.Bar(
            x=k, y=pmf,
            name=f"Poisson PMF (μ={mu:.3f})",
            marker=dict(color="rgba(128, 0, 128, 0.6)"),  # purple, 60% opacity
            hovertemplate="k=%{x}<br>P(N=k)=%{y:.4f}<extra></extra>"
        ))

        # Overlay line + markers — solid purple
        fig.add_trace(go.Scatter(
            x=k, y=pmf,
            mode="lines+markers",
            line=dict(color="rgb(128, 0, 128)", width=2),
            marker=dict(color="rgb(128, 0, 128)", size=5),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Vertical dashed mean line
        fig.add_vline(
            x=mu,
            line_dash="dash",
            line_color="purple",
            annotation_text=f"mean μ = {mu:.3f}",
            annotation_position="top right"
        )

        fig.update_layout(
            title=f"Poisson counts within period (λ={lam}, t={period}, μ=λt={mu:.3f})",
            xaxis_title="k (number of events)",
            yaxis_title="P(N = k)",
            bargap=0.15
        )
        fig.update_yaxes(rangemode="tozero")

        return fig

    poisson_fig = make_poisson_plot(lam=slider.value, period=poisson_days_slider.value)

    mo.ui.plotly(poisson_fig)
    return (make_poisson_plot,)


@app.cell(hide_code=True)
def _(mo):
    exp_slider = mo.ui.slider(start=0.01, stop=1, step=0.01, value=0.1, label="Lambda")
    gamma_slider = mo.ui.slider(start=1, stop=100, step=1, value=1, label="Events")
    poisson_slider = mo.ui.slider(start=1, stop=100, step=1, value=1, label="Time")

    mo.vstack([exp_slider, gamma_slider, poisson_slider])
    return exp_slider, gamma_slider, poisson_slider


@app.cell(hide_code=True)
def _(
    exp_slider,
    gamma_slider,
    make_exponential_plot,
    make_gamma_plot,
    make_poisson_plot,
    mo,
    poisson_slider,
):

    figures = [
        make_exponential_plot(exp_slider.value), 
        make_gamma_plot(lam=exp_slider.value, k=gamma_slider.value), 
        make_poisson_plot(lam=exp_slider.value, period=poisson_slider.value)
    ]

    for f in figures:
        f.update_layout(
            height=350, 
            width=450, 
            margin=dict(l=20,r=20,t=40,b=20), 
            showlegend=False
        )

    mo.hstack(figures, widths="equal")
    return


@app.cell
def _():
    # TODO:
    # Make sampling
    # Make one for linear regression 
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
