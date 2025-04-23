from flask import Flask, render_template, request, jsonify
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Força uso de backend sem Tkinter
import matplotlib.pyplot as plt

import io
import base64
import control as ctl
import json 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pagina2')
def pagina2():
    return render_template('pagina2.html')

@app.route('/pagina3')
def pagina3():
    return render_template('pagina3.html')

@app.route('/pagina4')
def pagina4():
    return render_template('pagina4.html')

@app.route('/atualizar', methods=['POST'])
def atualizar():
    data = request.get_json()
    tipo = data.get("tipo", "Caso 1")

    # Recebe os parâmetros da planta
    p1_1 = float(data.get("p1_1", -1))
    z_1 = float(data.get("z_1", 0))
    p1_2 = float(data.get("p1_2", -1))
    p2_2 = float(data.get("p2_2", -2))
    z_2 = float(data.get("z_2", 0))

    # Define a função de transferência da planta
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

    # Define as funções de transferência
    latex_planta = f"\\[ G_{{planta}}(s) = \\frac{{{np.poly1d(num)}}}{{{np.poly1d(den)}}} \\]"
    latex_controlador = "\\[ G_{{controlador}}(s) = \\frac{{s + 1}}{{s + 2}} \\]"
    latex_open = "\\[ G_{{open}}(s) = G_{{planta}}(s) \\cdot G_{{controlador}}(s) \\]"
    latex_closed = "\\[ G_{{closed}}(s) = \\frac{{G_{{open}}(s)}}{{1 + G_{{open}}(s)}} \\]"

    # Gera os gráficos
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Diagrama de Polos e Zeros
    zeros = np.roots(num)
    poles = np.roots(den)
    axs[0].scatter(np.real(zeros), np.imag(zeros), label="Zeros", color="blue")
    axs[0].scatter(np.real(poles), np.imag(poles), label="Polos", color="red")
    axs[0].set_title("Diagrama de Polos e Zeros")
    axs[0].grid(True)
    axs[0].legend()

    # Resposta ao Degrau (Malha Aberta)
    T, yout = ctl.step_response(G)
    axs[1].plot(T, yout)
    axs[1].set_title("Resposta ao Degrau (Malha Aberta)")
    axs[1].grid(True)

    # Salva os gráficos em base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "latex_planta": latex_planta,
        "latex_controlador": latex_controlador,
        "latex_open": latex_open,
        "latex_closed": latex_closed,
        "plot_url": f"data:image/png;base64,{plot_base64}"
    })

@app.route('/atualizar_bode', methods=['POST'])
def atualizar_bode():
    data = request.get_json()

    # Recebe os parâmetros da planta
    p1_1 = float(data.get("p1_1", -1))
    z_1 = float(data.get("z_1", 0))

    # Define a função de transferência da planta
    num = np.poly([z_1])
    den = np.poly([p1_1])
    G = ctl.tf(num, den)

    # Calcula os dados para o diagrama de Bode
    omega = np.logspace(-2, 2, 500)  # Frequência em rad/s
    mag, phase, omega = ctl.bode(G, omega=omega, dB=True, plot=False)

    # Gera o gráfico de Bode manualmente
    fig_bode, (mag_ax, phase_ax) = plt.subplots(2, 1, figsize=(10, 6))

    # Plota magnitude
    mag_ax.semilogx(omega, 20 * np.log10(mag))
    mag_ax.set_title("Diagrama de Bode - Magnitude")
    mag_ax.set_ylabel("Magnitude (dB)")
    mag_ax.set_xlabel("Frequência (rad/s)")
    mag_ax.grid(which="both", linestyle="--", linewidth=0.5)

    # Plota fase
    phase_ax.semilogx(omega, np.degrees(phase))
    phase_ax.set_title("Diagrama de Bode - Fase")
    phase_ax.set_ylabel("Fase (graus)")
    phase_ax.set_xlabel("Frequência (rad/s)")
    phase_ax.grid(which="both", linestyle="--", linewidth=0.5)

    # Salva o gráfico em um buffer
    plt.tight_layout()
    buf_bode = io.BytesIO()
    plt.savefig(buf_bode, format='png')
    plt.close(fig_bode)
    buf_bode.seek(0)
    bode_base64 = base64.b64encode(buf_bode.read()).decode('utf-8')

    return jsonify({
        "bode_url": f"data:image/png;base64,{bode_base64}"
    })

@app.route('/atualizar_nyquist', methods=['POST'])
def atualizar_nyquist():
    data = request.get_json()

    # Recebe os parâmetros da planta
    p1_1 = float(data.get("p1_1", -1))
    z_1 = float(data.get("z_1", 0))

    # Define a função de transferência da planta
    num = np.poly([z_1])
    den = np.poly([p1_1])
    G = ctl.tf(num, den)

    # Gera o gráfico de Nyquist
    fig_nyquist = plt.figure(figsize=(6, 6))
    ctl.nyquist(G, omega=np.logspace(-2, 2, 500))
    plt.title("Diagrama de Nyquist")
    plt.grid(which="both", linestyle="--", linewidth=0.5)

    # Salva o gráfico em um buffer
    buf_nyquist = io.BytesIO()
    plt.savefig(buf_nyquist, format='png')
    plt.close(fig_nyquist)
    buf_nyquist.seek(0)
    nyquist_base64 = base64.b64encode(buf_nyquist.read()).decode('utf-8')

    return jsonify({
        "nyquist_url": f"data:image/png;base64,{nyquist_base64}"
    })

@app.route('/atualizar_pagina4', methods=['POST'])
def atualizar_pagina4():
    data = request.get_json()
    p1_1 = float(data.get("p1_1", -1))
    z_1 = float(data.get("z_1", 0))

    # Define a função de transferência da planta
    num = np.poly([z_1])
    den = np.poly([p1_1])
    G = ctl.tf(num, den)

    # Dados para o gráfico de Polos e Zeros
    zeros = np.roots(num)
    poles = np.roots(den)
    plot_pz_data = {
        "data": [
            {"x": np.real(zeros).tolist(), "y": np.imag(zeros).tolist(), "mode": "markers", "name": "Zeros"},
            {"x": np.real(poles).tolist(), "y": np.imag(poles).tolist(), "mode": "markers", "name": "Polos"}
        ],
        "layout": {"title": "Diagrama de Polos e Zeros", "xaxis": {"title": "Re"}, "yaxis": {"title": "Im"}}
    }

    # Dados para o gráfico de Resposta ao Degrau
    T, yout = ctl.step_response(G)
    plot_open_data = {
        "data": [
            {"x": T.tolist(), "y": yout.tolist(), "mode": "lines", "name": "Resposta ao Degrau"}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Aberta)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    return jsonify({
        "latex_planta_monomial": f"\\[ G(s) = \\frac{{{np.poly1d(num)}}}{{{np.poly1d(den)}}} \\]",
        "latex_planta_polynomial": f"\\[ G(s) = \\frac{{{' + '.join(map(str, num))}}}{{{' + '.join(map(str, den))}}} \\]",
        "plot_pz_data": plot_pz_data,
        "plot_open_data": plot_open_data
    })

if __name__ == '__main__':
    app.run(debug=True)
