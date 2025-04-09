from flask import Flask, render_template, request, jsonify
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 👈 Força uso de backend sem Tkinter
import matplotlib.pyplot as plt

import io
import base64
import control as ctl

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/atualizar', methods=['POST'])
def atualizar():
    data = request.get_json()
    tipo = data.get("tipo", "Caso 1")

    # Recebe os parâmetros da planta
    params = {
        "p1_1": float(data.get("p1_1", -1) or -1),
        "z_1": float(data.get("z_1", 0) or 0),
        "p1_2": float(data.get("p1_2", -1) or -1),
        "p2_2": float(data.get("p2_2", -2) or -2),
        "z_2": float(data.get("z_2", 0) or 0),
    }

    # Recebe os parâmetros do controlador
    polo_controlador = float(data.get("polo_controlador", -1) or -1)
    zero_controlador = float(data.get("zero_controlador", 0) or 0)

    # Define a função de transferência da planta
    if tipo == "Caso 1":
        num = np.poly([params["z_1"]])
        den = np.poly([params["p1_1"]])
    elif tipo == "Caso 2":
        num = np.poly([params["z_2"]])
        den = np.poly([params["p1_2"], params["p2_2"]])
    else:
        num = [1]
        den = [1]

    # Normaliza a FT da planta
    N0 = np.polyval(num, 0)
    D0 = np.polyval(den, 0)
    if abs(N0) > 1e-8:
        K = D0 / N0
        num = K * num

    G = ctl.tf(num, den)

    # Define a função de transferência do controlador
    num_c = np.poly([zero_controlador])
    den_c = np.poly([polo_controlador])

    # Normaliza a FT do controlador
    N0_c = np.polyval(num_c, 0)
    D0_c = np.polyval(den_c, 0)
    if abs(N0_c) > 1e-8:
        K_c = D0_c / N0_c
        num_c = K_c * num_c

    C = ctl.tf(num_c, den_c)

    # Malha aberta e fechada
    G_open = C * G  # Malha aberta
    G_closed = ctl.feedback(G_open)  # Malha fechada (feedback unitário)

    # Formata para ZPK
    def zpk_to_latex(z, p, k):
        def format_roots(roots):
            return " ".join([f"(s - {root:.2f})" for root in roots]) if len(roots) > 0 else "1"
        zeros = format_roots(z)
        poles = format_roots(p)
        gain = f"{k:.2f}"
        return f"\\[ G(s) = {gain} \\frac{{{zeros}}}{{{poles}}} \\]"

    # Obtém ZPK para as malhas
    z_open, p_open, k_open = ctl.zeros(G_open), ctl.poles(G_open), ctl.dcgain(G_open)
    z_closed, p_closed, k_closed = ctl.zeros(G_closed), ctl.poles(G_closed), ctl.dcgain(G_closed)

    latex_open = zpk_to_latex(z_open, p_open, k_open)
    latex_closed = zpk_to_latex(z_closed, p_closed, k_closed)

    # Funções de transferência da planta e do controlador
    latex_planta = f"\\[ G_{{planta}}(s) = \\frac{{{np.poly1d(G.num[0][0])}}}{{{np.poly1d(G.den[0][0])}}} \\]"
    latex_controlador = f"\\[ G_{{controlador}}(s) = \\frac{{{np.poly1d(C.num[0][0])}}}{{{np.poly1d(C.den[0][0])}}} \\]"

    # Gera os gráficos
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    zeros = np.roots(G_closed.num[0][0])
    polos = np.roots(G_closed.den[0][0])

    axs[0].plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros')
    axs[0].plot(np.real(polos), np.imag(polos), 'x', label='Polos')
    axs[0].set_title("Polos e Zeros (Malha Fechada)")
    axs[0].grid(True)
    axs[0].legend()

    try:
        T_open, yout_open = ctl.step_response(G_open)
        axs[1].plot(T_open, yout_open)
        axs[1].set_title("Resposta ao Degrau (Malha Aberta)")
        axs[1].grid(True)

        T_closed, yout_closed = ctl.step_response(G_closed)
        axs[2].plot(T_closed, yout_closed)
        axs[2].set_title("Resposta ao Degrau (Malha Fechada)")
        axs[2].grid(True)
    except Exception as e:
        axs[1].text(0.5, 0.5, f"Erro: {e}", ha='center', va='center')
        axs[2].text(0.5, 0.5, f"Erro: {e}", ha='center', va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({
        "latex_planta": latex_planta,
        "latex_controlador": latex_controlador,
        "latex_open": latex_open,
        "latex_closed": latex_closed,
        "plot_url": f"data:image/png;base64,{plot_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True)
