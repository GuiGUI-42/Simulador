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

    # Recebe os parâmetros
    params = {
        "p1_1": float(data.get("p1_1", -1) or -1),
        "z_1": float(data.get("z_1", 0) or 0),
        "p1_2": float(data.get("p1_2", -1) or -1),
        "p2_2": float(data.get("p2_2", -2) or -2),
        "z_2": float(data.get("z_2", 0) or 0),
    }

    # Define a função de transferência
    if tipo == "Caso 1":
        num = np.poly([params["z_1"]])
        den = np.poly([params["p1_1"]])
    elif tipo == "Caso 2":
        num = np.poly([params["z_2"]])
        den = np.poly([params["p1_2"], params["p2_2"]])
    else:
        num = [1]
        den = [1]

    # Normaliza a FT
    N0 = np.polyval(num, 0)
    D0 = np.polyval(den, 0)
    if abs(N0) > 1e-8:
        K = D0 / N0
        num = K * num

    G = ctl.tf(num, den)

    # Faça o mesmo para um controlador C
    num_C = [1]
    den_C = [1, 0]
    C = ctl.tf(num_C, den_C)
    H = ctl.series(G, C)


    # Formata para LaTeX
    def poly_to_latex(p):
        terms = []
        n = len(p)
        for i, coef in enumerate(p):
            exp = n - i - 1
            if abs(coef) < 1e-8:
                continue
            s_term = f"s^{{{exp}}}" if exp > 1 else "s" if exp == 1 else ""
            coef_str = f"{coef:.2f}" if abs(coef - 1) > 1e-8 or exp == 0 else ""
            terms.append(f"{coef_str}{s_term}")
        return " + ".join(terms) if terms else "0"

    latex_func = f"\\[ G(s) = \\frac{{{poly_to_latex(num)}}}{{{poly_to_latex(den)}}} \\]"

    # Gera o gráfico
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    zeros = np.roots(num)
    polos = np.roots(den)

    axs[0].plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros')
    axs[0].plot(np.real(polos), np.imag(polos), 'x', label='Polos')
    axs[0].set_title("Polos e Zeros")
    axs[0].grid(True)
    axs[0].legend()

    try:
        T, yout = ctl.step_response(G)
        axs[1].plot(T, yout)
        axs[1].set_title("Resposta ao Degrau")
        axs[1].grid(True)
    except Exception as e:
        axs[1].text(0.5, 0.5, f"Erro: {e}", ha='center', va='center')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return jsonify({
        "latex_funcao": latex_func,
        "plot_url": f"data:image/png;base64,{plot_base64}"
    })

if __name__ == '__main__':
    app.run(debug=True)
