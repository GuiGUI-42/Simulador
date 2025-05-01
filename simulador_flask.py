from flask import Flask, render_template, request, jsonify
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Força uso de backend sem Tkinter
import matplotlib.pyplot as plt

import io
import base64
import control as ctl
import json 
import smtplib
from email.mime.text import MIMEText

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

    t_perturb = float(data.get("t_perturb", 20))
    amp_perturb = float(data.get("amp_perturb", 0.5))

    # Resposta ao degrau (sem perturbação)
    T, yout_step = ctl.step_response(G)

    # Resposta ao degrau + perturbação
    T_long = np.linspace(0, 50, 1000)
    u = np.ones_like(T_long)
    u[T_long >= t_perturb] += amp_perturb
    T_resp, yout_perturb = ctl.forced_response(G, T_long, u)

    return jsonify({
        "latex_planta": latex_planta,
        "latex_controlador": latex_controlador,
        "latex_open": latex_open,
        "latex_closed": latex_closed,
        "plot_data": {
            "T": T_long.tolist(),
            "yout_step": np.interp(T_long, T, yout_step).tolist(),
            "yout_perturb": yout_perturb.tolist(),
            "t_perturb": t_perturb,
            "amp_perturb": amp_perturb
        }
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
    polos_planta = data.get("polos_planta", [-1])
    zeros_planta = data.get("zeros_planta", [0])
    polos_controlador = data.get("polos_controlador", [-1])
    zeros_controlador = data.get("zeros_controlador", [0])

    # Filtra zeros realmente diferentes de zero
    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta)

    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    num_controlador = np.poly(zeros_controlador_filtrados) if zeros_controlador_filtrados else np.array([1.0])
    den_controlador = np.poly(polos_controlador)

    G_planta = ctl.tf(num_planta, den_planta)
    G_controlador = ctl.tf(num_controlador, den_controlador)

    G_open = G_planta
    G_closed = ctl.feedback(G_planta * G_controlador)

    # Arredonda os coeficientes para três casas decimais
    num_planta_rounded = np.round(num_planta, 3)
    den_planta_rounded = np.round(den_planta, 3)
    num_controlador_rounded = np.round(num_controlador, 3)
    den_controlador_rounded = np.round(den_controlador, 3)

    # Formata as funções de transferência com os coeficientes arredondados
    latex_planta_monomial = f"\\[ G(s) = \\frac{{{np.poly1d(num_planta_rounded, variable='s')}}}{{{np.poly1d(den_planta_rounded, variable='s')}}} \\]"
    latex_controlador = f"\\[ G_c(s) = \\frac{{{np.poly1d(num_controlador_rounded, variable='s')}}}{{{np.poly1d(den_controlador_rounded, variable='s')}}} \\ ]"

    # Dados para os gráficos
    zeros = np.roots(num_planta)
    poles = np.roots(den_planta)
    plot_pz_data = {
        "data": [
            {"x": np.real(zeros).tolist(), "y": np.imag(zeros).tolist(), "mode": "markers", "name": "Zeros"},
            {"x": np.real(poles).tolist(), "y": np.imag(poles).tolist(), "mode": "markers", "name": "Polos"}
        ],
        "layout": {"title": "Diagrama de Polos e Zeros", "xaxis": {"title": "Re"}, "yaxis": {"title": "Im"}}
    }

    # Resposta ao degrau (Malha Aberta - apenas planta)
    T_open, yout_open = ctl.step_response(G_open)
    plot_open_data = {
        "data": [
            {"x": T_open.tolist(), "y": yout_open.tolist(), "mode": "lines", "name": "Resposta ao Degrau (Malha Aberta)"}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Aberta)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    # Resposta ao degrau (Malha Fechada - planta e controlador)
    T_closed, yout_closed = ctl.step_response(G_closed)
    plot_closed_data = {
        "data": [
            {"x": T_closed.tolist(), "y": yout_closed.tolist(), "mode": "lines", "name": "Resposta ao Degrau (Malha Fechada)"}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Fechada)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    # Gere as strings LaTeX para as FTs
    ordem_den_planta = len(den_planta) - 1
    # Ordem do numerador: conta quantos zeros são diferentes de zero
    ordem_num_planta = np.sum(np.abs(zeros_planta) > 1e-8)
    if ordem_num_planta == 0:
        ordem_num_planta = 0 if np.allclose(num_planta, [1]) else len(num_planta) - 1

    ordem_controlador = len(den_controlador) - 1
    ordem_num_controlador = np.sum(np.abs(zeros_controlador) > 1e-8)
    if ordem_num_controlador == 0:
        ordem_num_controlador = 0 if np.allclose(num_controlador, [1]) else len(num_controlador) - 1

    latex_planta_polinomial = (
        f"\\[ \\text{{Ordem: }} {ordem_den_planta} \\qquad "
        f"G(s) = \\frac{{{latex_poly(num_planta, 's')}}}{{{latex_poly(den_planta, 's')}}} \\]"
    )
    latex_planta_fatorada = (
        f"\\[ \\text{{Ordem: }} {ordem_den_planta} \\qquad "
        f"G(s) = \\frac{{{latex_factored(zeros_planta, 's')}}}{{{latex_factored(polos_planta, 's')}}} \\]"
    )
    latex_controlador_polinomial = (
        f"\\[ \\text{{Ordem: }} {ordem_controlador} \\qquad "
        f"G_c(s) = \\frac{{{latex_poly(num_controlador, 's')}}}{{{latex_poly(den_controlador, 's')}}} \\]"
    )
    latex_controlador_fatorada = (
        f"\\[ \\text{{Ordem: }} {ordem_controlador} \\qquad "
        f"G_c(s) = \\frac{{{latex_factored(zeros_controlador, 's')}}}{{{latex_factored(polos_controlador, 's')}}} \\]"
    )

    latex_planta_monic = (
        f"\\[ \\text{{Ordem: }} {ordem_den_planta} "
        f"G(s) = \\frac{{{latex_monic(num_planta, 's')}}}{{{latex_monic(den_planta, 's')}}} \\]"
    )
    latex_controlador_monic = (
        f"\\[ \\text{{Ordem: }} {ordem_controlador} "
        f"G_c(s) = \\frac{{{latex_monic(num_controlador, 's')}}}{{{latex_monic(den_controlador, 's')}}} \\]"
    )

    return jsonify({
        "latex_planta_monomial": latex_planta_monomial,
        "latex_controlador": latex_controlador,
        "plot_pz_data": plot_pz_data,
        "plot_open_data": plot_open_data,
        "plot_closed_data": plot_closed_data,
        "latex_planta_polinomial": latex_planta_polinomial,
        "latex_planta_monic": latex_planta_monic,
        "latex_planta_fatorada": latex_planta_fatorada,
        "latex_controlador_polinomial": latex_controlador_polinomial,
        "latex_controlador_monic": latex_controlador_monic,
        "latex_controlador_fatorada": latex_controlador_fatorada
    })

@app.route('/enviar_feedback', methods=['POST'])
def enviar_feedback():
    data = request.get_json()
    texto = data.get('feedback', '')
    # Configurações do seu e-mail
    remetente = 'syscoufsc@gmail.com'
    senha = 'jvey wybi xcnm znko'  # Use senha de app para Gmail
    destinatario = 'syscoufsc@gmail.com'  # Ou outro e-mail de destino

    msg = MIMEText(texto)
    msg['Subject'] = 'Feedback do Simulador'
    msg['From'] = remetente
    msg['To'] = destinatario

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(remetente, senha)
            server.sendmail(remetente, destinatario, msg.as_string())
        return jsonify({'status': 'ok'})
    except Exception as e:
        print("Erro ao enviar feedback:", e)
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 500

def latex_poly(coeffs, var='s'):
    coeffs = np.array(coeffs, dtype=float)
    # Remove zeros à direita (constante no início)
    coeffs = np.trim_zeros(coeffs, 'f')
    # Se todos os zeros são zero, o numerador vira [1, 0, 0, ...]
    # Queremos mostrar "1" se todos os coeficientes, exceto o primeiro, são zero
    if len(coeffs) > 1 and np.allclose(coeffs[1:], 0) and np.isclose(coeffs[0], 1):
        return "1"
    if len(coeffs) == 1 and np.isclose(coeffs[0], 1):
        return "1"
    ordem = len(coeffs) - 1
    termos = []
    for i, c in enumerate(coeffs):
        pot = ordem - i
        if abs(c) < 1e-10:
            continue
        c_str = f"{c:.3g}" if abs(c) != 1 or pot == 0 else ("-" if c == -1 else "")
        if pot > 1:
            termos.append(f"{c_str}{var}^{pot}")
        elif pot == 1:
            termos.append(f"{c_str}{var}")
        else:
            termos.append(f"{c_str}")
    return " + ".join(termos).replace("+ -", "- ")

def latex_monic(coeffs, var='s'):
    if abs(coeffs[0]) < 1e-10:
        return latex_poly(coeffs, var)
    norm = coeffs / coeffs[0]
    return latex_poly(norm, var)

def latex_factored(roots, var='s'):
    if len(roots) == 0 or np.allclose(roots, 0):
        return "1"
    termos = []
    for r in roots:
        if abs(r) < 1e-10:
            termos.append(f"{var}")
        elif r < 0:
            termos.append(f"({var} {r:+.3g})")
        else:
            termos.append(f"({var} - {abs(r):.3g})")
    return "".join(termos)

if __name__ == '__main__':
    app.run(debug=True)
