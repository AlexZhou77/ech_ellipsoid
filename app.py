import numpy as np
import plotly.graph_objects as go
from dash import Dash, Patch, State, dcc, html, Input, Output

from ellipsoid import Ellipsoid

a = 1.0
b = np.sqrt(2.0)
ellipsoid = Ellipsoid(a, b)

TRACE_INDEX_SURFACE = 0
TRACE_INDEX_REEB = 1
TRACE_INDEX_GAMMA1 = 2
TRACE_INDEX_GAMMA2 = 3
TRACE_INDEX_CONTACT_PLANES1 = 4
TRACE_INDEX_CONTACT_PLANE_EDGES1 = 5
TRACE_INDEX_CONTACT_PLANES2 = 6
TRACE_INDEX_CONTACT_PLANE_EDGES2 = 7
TRACE_INDEX_BRAID1 = 8
TRACE_INDEX_BRAID2 = 9

MOVING_CIRCLE_SAMPLES = 360
BRAID_SAMPLES = 500
DEFAULT_M1 = 1
DEFAULT_M2 = 1
DEFAULT_S = 2.0


def empty_trace_xyz():
    return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])


def sanitize_multiplicity(value, default=1):
    if value in (None, ""):
        return default
    return max(1, int(value))


def make_figure(eta, s=DEFAULT_S, m1=DEFAULT_M1, m2=DEFAULT_M2, show_trajectory=False, show_gamma1=True, show_gamma2=True, show_contact_planes1=False, show_contact_planes2=False, show_braid1=False, show_braid2=False):
    fig = go.Figure()

    m1 = sanitize_multiplicity(m1, DEFAULT_M1)
    m2 = sanitize_multiplicity(m2, DEFAULT_M2)

    # Torus leaf
    X, Y, Z = ellipsoid.torus_surface(eta, n1=140, n2=140)

    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.42, colorscale=[[0.0, "#f2d9f6"], [1.0, "#f2cdf6"]], showscale=False, name="Torus leaf", hoverinfo="skip"))

    # Reeb trajectory
    Xt, Yt, Zt = ellipsoid.reeb_trajectory(eta, theta1_0=0.0, theta2_0=0.0, T=12.0, N=2500)

    fig.add_trace(go.Scatter3d(x=Xt, y=Yt, z=Zt, mode="lines", line=dict(width=4, color="#be185d"), name="Reeb trajectory", visible=show_trajectory))

    # gamma_1
    X1, Y1, Z1 = ellipsoid.gamma1(n=800)

    fig.add_trace(go.Scatter3d(x=X1, y=Y1, z=Z1, mode="lines", line=dict(width=8, color="#2f80ed"), name="gamma_1: z2 = 0", visible=show_gamma1))

    # gamma_2
    X2, Y2, Z2 = ellipsoid.gamma2(n=800)

    fig.add_trace(go.Scatter3d(x=X2, y=Y2, z=Z2, mode="lines", line=dict(width=8, color="#ff9aa2"), name="gamma_2: z1 = 0", visible=show_gamma2))

    # Sampled contact planes along gamma_1
    plane_x, plane_y, plane_z, plane_i, plane_j, plane_k, edge_x, edge_y, edge_z = ellipsoid.gamma1_contact_plane_patches()

    fig.add_trace(go.Mesh3d(x=plane_x, y=plane_y, z=plane_z, i=plane_i, j=plane_j, k=plane_k, color="#34d399", opacity=0.34, flatshading=True, name="Contact planes along gamma_1", visible=show_contact_planes1, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines", line=dict(width=3, color="#059669"), name="Contact plane edges", visible=show_contact_planes1, hoverinfo="skip"))

    # Sampled contact planes along gamma_2
    plane2_x, plane2_y, plane2_z, plane2_i, plane2_j, plane2_k, edge2_x, edge2_y, edge2_z = ellipsoid.gamma2_contact_plane_patches()

    fig.add_trace(go.Mesh3d(x=plane2_x, y=plane2_y, z=plane2_z, i=plane2_i, j=plane2_j, k=plane2_k, color="#f9a8d4", opacity=0.34, flatshading=True, name="Contact planes along gamma_2", visible=show_contact_planes2, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=edge2_x, y=edge2_y, z=edge2_z, mode="lines", line=dict(width=3, color="#ec4899"), name="Contact plane edges gamma_2", visible=show_contact_planes2, hoverinfo="skip"))

    braid_branches = ellipsoid.braid_slice_branches(s, m1, m2, n=BRAID_SAMPLES)
    braid1 = braid_branches[0] if braid_branches else empty_trace_xyz()
    braid2 = braid_branches[1] if len(braid_branches) > 1 else empty_trace_xyz()

    fig.add_trace(go.Scatter3d(x=braid1[0], y=braid1[1], z=braid1[2], mode="lines", line=dict(width=9, color="#f97316"), name="Braid around gamma_1", visible=show_braid1 and len(braid_branches) >= 1))
    fig.add_trace(go.Scatter3d(x=braid2[0], y=braid2[1], z=braid2[2], mode="lines", line=dict(width=9, color="#f97316"), name="Braid around gamma_2", visible=show_braid2 and len(braid_branches) >= 2))

    fig.update_layout(
        title=dict(text="Boundary of E(a,b)", x=0.5, xanchor="center", font=dict(size=22)),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(title="X", showbackground=True, backgroundcolor="rgb(245,245,245)", gridcolor="white", zerolinecolor="white"),
            yaxis=dict(title="Y", showbackground=True, backgroundcolor="rgb(245,245,245)", gridcolor="white", zerolinecolor="white"),
            zaxis=dict(title="Z", showbackground=True, backgroundcolor="rgb(245,245,245)", gridcolor="white", zerolinecolor="white"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.1)),
        ),
        legend=dict(x=0.76, y=0.95, bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, font=dict(size=13)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        uirevision="ellipsoid-view",
    )

    return fig

# The Dash app layout and callbacks
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "height": "100vh", "display": "flex", "backgroundColor": "white"},
    children=[
        # Left panel
        html.Div(
            style={"width": "280px", "padding": "24px", "borderRight": "1px solid #ddd", "boxSizing": "border-box"},
            children=[
                html.H2("Instructions", style={"marginTop": "0", "fontSize": "24px"}),
                dcc.Markdown(
                    r"""
This tool visualizes the Reeb dynamics on the boundary $Y = \partial E(a,b)$ of an ellipsoid.

We use coordinates $(\eta, \theta_1, \theta_2)$ on $Y$ given by

$$
z_1 = \sqrt{a / \pi}\, \cos(\eta) e^{i\theta_1}, \qquad
z_2 = \sqrt{b / \pi}\, \sin(\eta) e^{i\theta_2},
$$

where $0 \leq \eta \leq \pi/2$ and $\theta_1, \theta_2$ are angular coordinates mod $2\pi$.

For each fixed $\eta$, the corresponding level set is a torus invariant under the Reeb flow.

The parameter $s$ is the $\mathbb{R}$-coordinate in the symplectization $\mathbb{R} \times Y$. The integers $m_1$ and $m_2$ are the multiplicities of $\gamma_1$ and $\gamma_2$ in the orbit set.
                    """,
                    mathjax=True,
                    style={"fontSize": "14px", "lineHeight": "1.45", "color": "#444"},
                ),
                html.Br(),
                html.Label("eta", style={"fontWeight": "bold", "fontSize": "14px"}),
                dcc.Slider(
                    id="eta-slider",
                    min=0.0,
                    max=np.pi / 2,
                    step=0.02,
                    value=0.0,
                    marks={
                        0: "0",
                        round(np.pi / 4, 2): "π/4",
                        round(np.pi / 2, 2): "π/2",
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                html.Label("s", style={"fontWeight": "bold", "fontSize": "14px"}),
                dcc.Slider(
                    id="s-slider",
                    min=0.02,
                    max=4.0,
                    step=0.02,
                    value=DEFAULT_S,
                    marks={1: "1", 2: "2", 3: "3", 4: "4"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                html.Label("m1", style={"fontWeight": "bold", "fontSize": "14px"}),
                dcc.Input(
                    id="m1-input",
                    type="number",
                    value=DEFAULT_M1,
                    min=1,
                    step=1,
                    debounce=True,
                    style={"width": "100%", "fontSize": "14px", "padding": "8px", "boxSizing": "border-box"},
                ),
                html.Br(),
                html.Label("m2", style={"fontWeight": "bold", "fontSize": "14px"}),
                dcc.Input(
                    id="m2-input",
                    type="number",
                    value=DEFAULT_M2,
                    min=1,
                    step=1,
                    debounce=True,
                    style={"width": "100%", "fontSize": "14px", "padding": "8px", "boxSizing": "border-box"},
                ),
                html.Br(),
                html.Br(),
                dcc.Checklist(
                    id="trajectory-toggle",
                    options=[{"label": " Show Reeb trajectory", "value": "show"}],
                    value=[],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="gamma1-toggle",
                    options=[{"label": " Show gamma_1", "value": "show"}],
                    value=["show"],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="braid1-toggle",
                    options=[{"label": " Braid around gamma_1", "value": "show"}],
                    value=[],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="gamma2-toggle",
                    options=[{"label": " Show gamma_2", "value": "show"}],
                    value=["show"],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="contact-planes1-toggle",
                    options=[{"label": " Show contact planes on gamma_1", "value": "show"}],
                    value=[],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="contact-planes2-toggle",
                    options=[{"label": " Show contact planes on gamma_2", "value": "show"}],
                    value=[],
                    style={"fontSize": "14px"},
                ),
                dcc.Checklist(
                    id="braid2-toggle",
                    options=[{"label": " Braid around gamma_2", "value": "show"}],
                    value=[],
                    style={"fontSize": "14px"},
                ),
                html.Br(),

            ],
        ),

        # Main graph
        html.Div(
            style={"flex": "1", "height": "100vh"},
            children=[
                dcc.Graph(
                    id="ellipsoid-graph",
                    figure=make_figure(0.0, DEFAULT_S, DEFAULT_M1, DEFAULT_M2, False, True, True, False, False, False, False),
                    style={"height": "100%", "width": "100%"},
                    config={"displayModeBar": True, "scrollZoom": True},
                )
            ],
        ),
    ],
)


@app.callback(
    Output("ellipsoid-graph", "figure", allow_duplicate=True),
    Input("eta-slider", "value"),
    prevent_initial_call=True,
)
def update_eta_geometry(eta):
    X, Y, Z = ellipsoid.torus_surface(eta, n1=140, n2=140)
    Xt, Yt, Zt = ellipsoid.reeb_trajectory(eta, theta1_0=0.0, theta2_0=0.0, T=12.0, N=2500)

    patched_figure = Patch()
    patched_figure["data"][TRACE_INDEX_SURFACE]["x"] = X.tolist()
    patched_figure["data"][TRACE_INDEX_SURFACE]["y"] = Y.tolist()
    patched_figure["data"][TRACE_INDEX_SURFACE]["z"] = Z.tolist()
    patched_figure["data"][TRACE_INDEX_REEB]["x"] = Xt.tolist()
    patched_figure["data"][TRACE_INDEX_REEB]["y"] = Yt.tolist()
    patched_figure["data"][TRACE_INDEX_REEB]["z"] = Zt.tolist()

    return patched_figure


@app.callback(
    Output("ellipsoid-graph", "figure", allow_duplicate=True),
    Input("trajectory-toggle", "value"),
    Input("gamma1-toggle", "value"),
    Input("gamma2-toggle", "value"),
    Input("contact-planes1-toggle", "value"),
    Input("contact-planes2-toggle", "value"),
    prevent_initial_call=True,
)
def update_visibility(trajectory_toggle, gamma1_toggle, gamma2_toggle, contact_planes1_toggle, contact_planes2_toggle):
    show_trajectory = "show" in trajectory_toggle
    show_gamma1 = "show" in gamma1_toggle
    show_gamma2 = "show" in gamma2_toggle
    show_contact_planes1 = "show" in contact_planes1_toggle
    show_contact_planes2 = "show" in contact_planes2_toggle

    patched_figure = Patch()
    patched_figure["data"][TRACE_INDEX_REEB]["visible"] = show_trajectory
    patched_figure["data"][TRACE_INDEX_GAMMA1]["visible"] = show_gamma1
    patched_figure["data"][TRACE_INDEX_GAMMA2]["visible"] = show_gamma2
    patched_figure["data"][TRACE_INDEX_CONTACT_PLANES1]["visible"] = show_contact_planes1
    patched_figure["data"][TRACE_INDEX_CONTACT_PLANE_EDGES1]["visible"] = show_contact_planes1
    patched_figure["data"][TRACE_INDEX_CONTACT_PLANES2]["visible"] = show_contact_planes2
    patched_figure["data"][TRACE_INDEX_CONTACT_PLANE_EDGES2]["visible"] = show_contact_planes2

    return patched_figure


@app.callback(
    Output("ellipsoid-graph", "figure", allow_duplicate=True),
    Input("s-slider", "value"),
    Input("m1-input", "value"),
    Input("m2-input", "value"),
    prevent_initial_call=True,
)
def update_braids_geometry(s, m1, m2):
    m1 = sanitize_multiplicity(m1, DEFAULT_M1)
    m2 = sanitize_multiplicity(m2, DEFAULT_M2)
    braid_branches = ellipsoid.braid_slice_branches(s, m1, m2, n=BRAID_SAMPLES)
    braid1 = braid_branches[0] if braid_branches else empty_trace_xyz()
    braid2 = braid_branches[1] if len(braid_branches) > 1 else empty_trace_xyz()

    patched_figure = Patch()
    patched_figure["data"][TRACE_INDEX_BRAID1]["x"] = braid1[0].tolist()
    patched_figure["data"][TRACE_INDEX_BRAID1]["y"] = braid1[1].tolist()
    patched_figure["data"][TRACE_INDEX_BRAID1]["z"] = braid1[2].tolist()
    patched_figure["data"][TRACE_INDEX_BRAID2]["x"] = braid2[0].tolist()
    patched_figure["data"][TRACE_INDEX_BRAID2]["y"] = braid2[1].tolist()
    patched_figure["data"][TRACE_INDEX_BRAID2]["z"] = braid2[2].tolist()
    return patched_figure


@app.callback(
    Output("ellipsoid-graph", "figure", allow_duplicate=True),
    Input("braid1-toggle", "value"),
    Input("braid2-toggle", "value"),
    State("s-slider", "value"),
    State("m1-input", "value"),
    State("m2-input", "value"),
    prevent_initial_call=True,
)
def update_braid_visibility(braid1_toggle, braid2_toggle, s, m1, m2):
    show_braid1 = "show" in braid1_toggle
    show_braid2 = "show" in braid2_toggle
    m1 = sanitize_multiplicity(m1, DEFAULT_M1)
    m2 = sanitize_multiplicity(m2, DEFAULT_M2)
    branch_count = len(ellipsoid.braid_etas(s, m1, m2))

    patched_figure = Patch()
    patched_figure["data"][TRACE_INDEX_BRAID1]["visible"] = show_braid1 and branch_count >= 1
    patched_figure["data"][TRACE_INDEX_BRAID2]["visible"] = show_braid2 and branch_count >= 2

    return patched_figure

if __name__ == "__main__":
    app.run(debug=False)