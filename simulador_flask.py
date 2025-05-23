from flask import Flask, render_template, request, jsonify, redirect
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Força uso de backend sem Tkinter
import matplotlib.pyplot as plt

plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False  # Garante que não tenta usar LaTeX
plt.rcParams['axes.formatter.use_mathtext'] = False

import io
import base64
import control as ctl
import json 
import smtplib
from email.mime.text import MIMEText
import scipy.signal

app = Flask(__name__)

@app.route('/')
def home():
    return redirect('/pagina4')

@app.route('/pagina2')
def pagina2():
    return render_template('pagina2.html')



@app.route('/pagina4')
def pagina4():
    return render_template('pagina4.html')

@app.route('/blocos')
def blocos():
    return render_template('bloco.html')

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
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]
    polos_controlador = [float(p) for p in data.get("polos_controlador", [-1])]
    zeros_controlador = [float(z) for z in data.get("zeros_controlador", [0])]
    ganho_controlador = float(data.get("ganho_controlador", 1.0))

    print("polos_planta:", polos_planta)
    print("zeros_planta:", zeros_planta)
    print("polos_controlador:", polos_controlador)
    print("zeros_controlador:", zeros_controlador)
    print("ganho_controlador:", ganho_controlador)

    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    polos_planta_filtrados = [p for p in polos_planta if abs(p) > 1e-8]
    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    polos_controlador_filtrados = [p for p in polos_controlador if abs(p) > 1e-8]

    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta_filtrados)
    num_controlador = np.poly(zeros_controlador_filtrados) if zeros_controlador_filtrados else np.array([1.0])
    num_controlador = ganho_controlador * num_controlador
    den_controlador = np.poly(polos_controlador_filtrados)

    num_open = np.polymul(num_planta, num_controlador)
    den_open = np.polymul(den_planta, den_controlador)
    G_open = ctl.tf(num_open, den_open)

    omega = np.logspace(-2, 2, 500)  # Frequências de 0.01 a 100 rad/s
    mag, phase, omega = ctl.bode(G_open, omega=omega, dB=True, plot=False)

    bode_data = {
        "data": [
            {
                "x": omega.tolist(),
                "y": 20 * np.log10(mag).tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Magnitude"
            },
            {
                "x": omega.tolist(),
                "y": np.degrees(phase).tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "Fase",
                "yaxis": "y2"
            }
        ],
        "layout": {
            "title": "Diagrama de Bode",
            "xaxis": {"title": "Frequência (rad/s)", "type": "log"},
            "yaxis": {"title": "Magnitude (dB)"},
            "yaxis2": {
                "title": "Fase (graus)",
                "overlaying": "y",
                "side": "right"
            },
            "legend": {"x": 0, "y": 1.1, "orientation": "h"}
        }
    }

    return jsonify({"bode_data": bode_data})

@app.route('/atualizar_nyquist', methods=['POST'])
def atualizar_nyquist():
    data = request.get_json()
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]
    polos_controlador = [float(p) for p in data.get("polos_controlador", [-1])]
    zeros_controlador = [float(z) for z in data.get("zeros_controlador", [0])]
    ganho_controlador = float(data.get("ganho_controlador", 1.0))

    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    polos_planta_filtrados = [p for p in polos_planta if abs(p) > 1e-8]
    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    polos_controlador_filtrados = [p for p in polos_controlador if abs(p) > 1e-8]

    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta_filtrados) if polos_planta_filtrados else np.array([1.0])
    num_controlador = np.poly(zeros_controlador_filtrados) if zeros_controlador_filtrados else np.array([1.0])
    num_controlador = ganho_controlador * num_controlador
    den_controlador = np.poly(polos_controlador_filtrados) if polos_controlador_filtrados else np.array([1.0])

    num_open = np.polymul(num_planta, num_controlador)
    den_open = np.polymul(den_planta, den_controlador)
    G_open = ctl.tf(num_open, den_open)

    omega = np.logspace(-2, 2, 500)
    _, H, _ = ctl.freqresp(G_open, omega)
    real = np.real(H[0]).tolist() if H.ndim == 3 else np.real(H).tolist()
    imag = np.imag(H[0]).tolist() if H.ndim == 3 else np.imag(H).tolist()

    nyquist_data = {
        "data": [
            {"x": real, "y": imag, "mode": "lines", "name": "Nyquist"},
            {"x": real, "y": [-i for i in imag], "mode": "lines", "name": "Nyquist (espelhado)", "line": {"dash": "dash"}}
        ],
        "layout": {
            "title": "Diagrama de Nyquist",
            "xaxis": {"title": "Eixo Real"},
            "yaxis": {"title": "Eixo Imaginário"},
            "showlegend": True,
            "yaxis_scaleanchor": "x",
            "yaxis_scaleratio": 1
        }
    }
    return jsonify({"nyquist_data": nyquist_data})

@app.route('/atualizar_pagina4', methods=['POST'])
def atualizar_pagina4():
    data = request.get_json()
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]
    polos_controlador = [float(p) for p in data.get("polos_controlador", [-1])]
    zeros_controlador = [float(z) for z in data.get("zeros_controlador", [0])]
    ganho_controlador = float(data.get("ganho_controlador", 1.0))

    t_perturb_fechada = float(data.get("t_perturb_fechada", 20))
    amp_perturb_fechada = float(data.get("amp_perturb_fechada", 0.5))

    # Polos e zeros da planta e do controlador (malha aberta)
    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    polos_planta_filtrados = [p for p in polos_planta if abs(p) > 1e-8]
    polos_controlador_filtrados = [p for p in polos_controlador if abs(p) > 1e-8]

    # Junta todos os polos e zeros da planta e do controlador
    zeros_aberta = zeros_planta_filtrados + zeros_controlador_filtrados
    polos_aberta = polos_planta_filtrados + polos_controlador_filtrados

    plot_pz_data = {
        "data": [
            {"x": np.real(zeros_aberta).tolist(), "y": np.imag(zeros_aberta).tolist(), "mode": "markers", "name": "Zeros"},
            {"x": np.real(polos_aberta).tolist(), "y": np.imag(polos_aberta).tolist(), "mode": "markers", "name": "Polos"}
        ],
        "layout": {
            "title": "Diagrama de Polos e Zeros (Malha Aberta)",
            "xaxis": {"title": "Re"},
            "yaxis": {"title": "Im"},
            "showlegend": True  # <-- Adicione esta linha
        }
    }

    # Filtra zeros realmente diferentes de zero
    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta)

    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    num_controlador = np.poly(zeros_controlador_filtrados) if zeros_controlador_filtrados else np.array([1.0])
    num_controlador = ganho_controlador * num_controlador  # Aplica o ganho
    den_controlador = np.poly(polos_controlador)

    G_planta = ctl.tf(num_planta, den_planta)
    G_controlador = ctl.tf(num_controlador, den_controlador)

    G_open = G_planta * G_controlador
    G_closed = ctl.feedback(G_planta * G_controlador)

    

    # Arredonda os coeficientes para três casas decimais
    num_planta_rounded = np.round(num_planta, 3)
    den_planta_rounded = np.round(den_planta, 3)
    num_controlador_rounded = np.round(num_controlador, 3)
    den_controlador_rounded = np.round(den_controlador, 3)

    # Formata as funções de transferência com os coeficientes arredondados
    latex_planta_monomial = f"\\[ G(s) = \\frac{{{np.poly1d(num_planta_rounded, variable='s')}}}{{{np.poly1d(den_planta_rounded, variable='s')}}} \\]"
    latex_controlador = f"\\[ G_c(s) = \\frac{{{np.poly1d(num_controlador_rounded, variable='s')}}}{{{np.poly1d(den_controlador_rounded, variable='s')}}} \\]"

    # Resposta ao degrau (Malha Aberta - apenas planta)
    T_open, yout_open = ctl.step_response(G_planta)  # <-- Use só a planta aqui!
    plot_open_data = {
        "data": [
            {"x": T_open.tolist(), "y": yout_open.tolist(), "mode": "lines", "name": "Resposta ao Degrau (Malha Aberta)"}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Aberta)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    # Resposta ao degrau (Malha Fechada - planta e controlador)
    T = np.linspace(0, 50, 1000)
    # Resposta sem perturbação
    _, yout_closed = ctl.forced_response(G_closed, T, np.ones_like(T))
    # Resposta com perturbação
    u_perturb = np.ones_like(T)
    u_perturb[T >= t_perturb_fechada] += amp_perturb_fechada
    _, yout_perturb = ctl.forced_response(G_closed, T, u_perturb)

    # Resposta ao degrau da planta (malha aberta) para o mesmo vetor T
    _, yout_open_interp = ctl.forced_response(G_planta, T, np.ones_like(T))

    plot_closed_data = {
        "data": [
            {"x": T.tolist(), "y": yout_closed.tolist(), "mode": "lines", "name": "Sem Perturbação"},
            {"x": T.tolist(), "y": yout_perturb.tolist(), "mode": "lines", "name": "Com Perturbação"},
            {"x": T.tolist(), "y": yout_open_interp.tolist(), "mode": "lines", "name": "Malha Aberta", "line": {"dash": "dash"}}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Fechada)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    # Gere as strings LaTeX para as FTs
    latex_planta_polinomial = (
        f"\\[ G(s) = \\frac{{{latex_poly(num_planta, 's')}}}{{{latex_poly(den_planta, 's')}}} \\]"
    )
    latex_planta_fatorada = (
        f"\\[ G(s) = \\frac{{{latex_factored(zeros_planta, 's')}}}{{{latex_factored(polos_planta, 's')}}} \\]"
    )
    latex_controlador_polinomial = (
        f"\\[ G_c(s) = \\frac{{{latex_poly(num_controlador, 's')}}}{{{latex_poly(den_controlador, 's')}}} \\]"
    )
    latex_controlador_fatorada = (
        f"\\[ G_c(s) = \\frac{{{latex_factored(zeros_controlador, 's')}}}{{{latex_factored(polos_controlador, 's')}}} \\]"
    )

    latex_planta_monic = (
        f"\\[ G(s) = \\frac{{{latex_monic(num_planta, 's')}}}{{{latex_monic(den_planta, 's')}}} \\]"
    )
    latex_controlador_monic = (
        f"\\[ G_c(s) = \\frac{{{latex_monic(num_controlador, 's')}}}{{{latex_monic(den_controlador, 's')}}} \\ ]"
    )

    # Use os zeros/polos do input diretamente para a fração parcial
    num_planta_parcial = np.poly(zeros_planta) if zeros_planta else np.array([1.0])
    den_planta_parcial = np.poly(polos_planta) if polos_planta else np.array([1.0])
    num_controlador_parcial = ganho_controlador * (np.poly(zeros_controlador) if zeros_controlador else np.array([1.0]))
    den_controlador_parcial = np.poly(polos_controlador) if polos_controlador else np.array([1.0])

    latex_planta_parcial = (
        f"\\[ G(s) = {latex_partial_fraction(num_planta_parcial, den_planta_parcial, 's')} \\]"
    )
    latex_controlador_parcial = (
        f"\\[ G_c(s) = {latex_partial_fraction(num_controlador_parcial, den_controlador_parcial, 's')} \\]"
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
        "latex_controlador_fatorada": latex_controlador_fatorada,
        "latex_planta_parcial": latex_planta_parcial,
        "latex_controlador_parcial": latex_controlador_parcial
    })

@app.route('/atualizar_pagina2', methods=['POST'])
def atualizar_pagina2():
    data = request.get_json()
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]

    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta)

    G_planta = ctl.tf(num_planta, den_planta)

    # Polos e zeros
    zeros = np.roots(num_planta)
    poles = np.roots(den_planta)
    plot_pz_data = {
        "data": [
            {"x": np.real(zeros).tolist(), "y": np.imag(zeros).tolist(), "mode": "markers", "name": "Zeros"},
            {"x": np.real(poles).tolist(), "y": np.imag(poles).tolist(), "mode": "markers", "name": "Polos"}
        ],
        "layout": {"title": "Diagrama de Polos e Zeros", "xaxis": {"title": "Re"}, "yaxis": {"title": "Im"}}
    }

    # Resposta ao degrau (Malha Aberta)
    T_open, yout_open = ctl.step_response(G_planta)
    plot_open_data = {
        "data": [
            {"x": T_open.tolist(), "y": yout_open.tolist(), "mode": "lines", "name": "Resposta ao Degrau (Malha Aberta)"}
        ],
        "layout": {"title": "Resposta ao Degrau (Malha Aberta)", "xaxis": {"title": "Tempo (s)"}, "yaxis": {"title": "Amplitude"}}
    }

    latex_planta_polinomial = (
        f"\\[ G(s) = \\frac{{{latex_poly(num_planta, 's')}}}{{{latex_poly(den_planta, 's')}}} \\]"
    )
    latex_planta_fatorada = (
        f"\\[ G(s) = \\frac{{{latex_factored(zeros_planta, 's')}}}{{{latex_factored(polos_planta, 's')}}} \\]"
    )

    return jsonify({
        "latex_planta_polinomial": latex_planta_polinomial,
        "latex_planta_fatorada": latex_planta_fatorada,
        "plot_pz_data": plot_pz_data,
        "plot_open_data": plot_open_data
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

def latex_partial_fraction(num, den, var='s'):
    num = np.trim_zeros(num, 'f')
    den = np.trim_zeros(den, 'f')
    if len(num) == 0 or len(den) == 0:
        return "0"
    r, p, k = scipy.signal.residue(num, den)
    termos = []
    for ri, pi in zip(r, p):
        ri = np.round(ri, 4)
        pi = np.round(pi, 4)
        if abs(ri.imag) < 1e-8:
            termos.append(f"\\frac{{{ri.real}}}{{{var} - ({pi.real})}}")
        else:
            termos.append(f"\\frac{{{ri}}}{{{var} - ({pi})}}")
    if k is not None and len(k) > 0:
        for i, ki in enumerate(k):
            if abs(ki) > 1e-8:
                termos.append(f"{ki:.4g}{var}^{len(k)-i-1}")
    return " + ".join(termos) if termos else "0"

@app.route('/nyquist_pagina2', methods=['POST'])
def nyquist_pagina2():
    data = request.get_json()
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]

    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta)

    G_planta = ctl.tf(num_planta, den_planta)

    fig, ax = plt.subplots(figsize=(7, 4))  # Apenas aumenta o tamanho, não altera escala
    ctl.nyquist_plot(G_planta, omega=np.logspace(-2, 2, 500), ax=ax, color='b')
    ax.set_title("Diagrama de Nyquist")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True)
    # Remova os set_xlim/set_ylim para manter a escala automática
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({"nyquist_img": img_base64})

@app.route('/nyquist_pagina4', methods=['POST'])
def nyquist_pagina4():
    data = request.get_json()
    polos_planta = [float(p) for p in data.get("polos_planta", [-1])]
    zeros_planta = [float(z) for z in data.get("zeros_planta", [0])]
    polos_controlador = [float(p) for p in data.get("polos_controlador", [-1])]
    zeros_controlador = [float(z) for z in data.get("zeros_controlador", [0])]
    ganho_controlador = float(data.get("ganho_controlador", 1.0))

    zeros_planta_filtrados = [z for z in zeros_planta if abs(z) > 1e-8]
    polos_planta_filtrados = [p for p in polos_planta if abs(p) > 1e-8]
    zeros_controlador_filtrados = [z for z in zeros_controlador if abs(z) > 1e-8]
    polos_controlador_filtrados = [p for p in polos_controlador if abs(p) > 1e-8]

    num_planta = np.poly(zeros_planta_filtrados) if zeros_planta_filtrados else np.array([1.0])
    den_planta = np.poly(polos_planta_filtrados) if polos_planta_filtrados else np.array([1.0])
    num_controlador = np.poly(zeros_controlador_filtrados) if zeros_controlador_filtrados else np.array([1.0])
    num_controlador = ganho_controlador * num_controlador
    den_controlador = np.poly(polos_controlador_filtrados) if polos_controlador_filtrados else np.array([1.0])

    num_open = np.polymul(num_planta, num_controlador)
    den_open = np.polymul(den_planta, den_controlador)
    G_open = ctl.tf(num_open, den_open)

    fig, ax = plt.subplots(figsize=(7, 4))  # Apenas aumenta o tamanho, não altera escala
    ctl.nyquist_plot(G_open, omega=np.logspace(-2, 2, 500), ax=ax, color='b')
    ax.set_title("Diagrama de Nyquist")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.grid(True)
    # Remova os set_xlim/set_ylim para manter a escala automática
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return jsonify({"nyquist_img": img_base64})

if __name__ == '__main__':
    #app.un(debug=True)
    app.run(debug=True, host='0.0.0.0')