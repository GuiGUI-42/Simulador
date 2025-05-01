import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import control as ctl

app = dash.Dash(__name__)

# Parâmetros iniciais
tipo_init = "Caso 1"
p1_1_init = -1
z_1_init = 0
p1_2_init = -1
p2_2_init = -2
z_2_init = 0
t_perturb_init = 20
amp_perturb_init = 0.5

def gerar_resposta(tipo, p1_1, z_1, p1_2, p2_2, z_2, t_perturb, amp_perturb):
    if tipo == "Caso 1":
        num = np.poly([z_1])
        den = np.poly([p1_1])
    elif tipo == "Caso 2":
        num = np.poly([z_2])
        den = np.poly([p1_2, p2_2])
    else:
        num = [1]
        den = [1]
    G = ctl.tf(num, den)
    T = np.linspace(0, 50, 1000)
    u = np.ones_like(T)
    u[T >= t_perturb] += amp_perturb
    _, yout_perturb = ctl.forced_response(G, T, u)
    _, yout_step = ctl.step_response(G, T)
    return T, yout_step, yout_perturb

app.layout = html.Div([
    html.H1("🛠️ Simulador de Controle (Dash)"),
    html.Div([
        html.Label("Tipo da Planta:"),
        dcc.Dropdown(
            id="tipo",
            options=[
                {"label": "Caso 1", "value": "Caso 1"},
                {"label": "Caso 2", "value": "Caso 2"}
            ],
            value=tipo_init,
            clearable=False,
            style={"width": "200px"}
        ),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.Div([
            html.Label("Polo:"),
            dcc.Slider(id="p1_1", min=-10, max=10, step=0.1, value=p1_1_init,
                       marks=None, tooltip={"placement": "bottom"}),
            dcc.Input(id="z_1", type="number", min=-10, max=10, step=0.1, value=z_1_init, style={"margin-left": "10px"}),
        ], id="caso1", style={"display": "block"}),
        html.Div([
            html.Label("Polo 1:"),
            dcc.Slider(id="p1_2", min=-10, max=10, step=0.1, value=p1_2_init,
                       marks=None, tooltip={"placement": "bottom"}),
            html.Label("Polo 2:"),
            dcc.Slider(id="p2_2", min=-10, max=10, step=0.1, value=p2_2_init,
                       marks=None, tooltip={"placement": "bottom"}),
            html.Label("Zero:"),
            dcc.Input(id="z_2", type="number", min=-10, max=10, step=0.1, value=z_2_init, style={"margin-left": "10px"}),
        ], id="caso2", style={"display": "none"}),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.H3("Perturbação"),
        html.Label("Tempo da Perturbação:"),
        dcc.Slider(id="t_perturb", min=0, max=50, step=0.1, value=t_perturb_init,
                   marks=None, tooltip={"placement": "bottom"}),
        html.Label("Amplitude da Perturbação:"),
        dcc.Slider(id="amp_perturb", min=-2, max=2, step=0.01, value=amp_perturb_init,
                   marks=None, tooltip={"placement": "bottom"}),
    ], style={"margin-bottom": "20px"}),

    html.Div([
        html.H3("Gráficos"),
        dcc.Graph(id="grafico", config={"editable": True}),
        html.Div("Arraste o ponto verde para mudar o tempo e a amplitude da perturbação.", style={"color": "green"})
    ])
])

@app.callback(
    Output("caso1", "style"),
    Output("caso2", "style"),
    Input("tipo", "value")
)
def mostrar_casos(tipo):
    if tipo == "Caso 1":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

@app.callback(
    Output("grafico", "figure"),
    Input("tipo", "value"),
    Input("p1_1", "value"),
    Input("z_1", "value"),
    Input("p1_2", "value"),
    Input("p2_2", "value"),
    Input("z_2", "value"),
    Input("t_perturb", "value"),
    Input("amp_perturb", "value"),
    Input("grafico", "relayoutData"),
    State("grafico", "figure")
)
def atualizar_grafico(tipo, p1_1, z_1, p1_2, p2_2, z_2, t_perturb, amp_perturb, relayoutData, fig):
    # Drag & drop do ponto de perturbação
    if relayoutData and fig and "x[2]" in relayoutData and "y[2]" in relayoutData:
        t_perturb = relayoutData["x[2]"]
        amp_perturb = relayoutData["y[2]"]
    T, yout_step, yout_perturb = gerar_resposta(tipo, p1_1, z_1, p1_2, p2_2, z_2, t_perturb, amp_perturb)
    idx = np.searchsorted(T, t_perturb)
    marker_y = yout_perturb[idx] if idx < len(yout_perturb) else yout_perturb[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=yout_step, mode="lines", name="Degrau"))
    fig.add_trace(go.Scatter(x=T, y=yout_perturb, mode="lines", name="Com Perturbação", line=dict(color="green")))
    fig.add_trace(go.Scatter(
        x=[t_perturb], y=[marker_y],
        mode="markers", marker=dict(color="lime", size=14, symbol="circle"),
        name="Perturbação"
    ))
    fig.update_layout(
        title="Resposta ao Degrau (Malha Aberta)",
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        dragmode="closest"
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)