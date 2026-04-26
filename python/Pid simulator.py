"""
=============================================================
  Simulador de Controlador PID - Resposta ao Degrau
=============================================================
  Autor: [Ruan Vitorino]
  Descrição:
    Simula um controlador PID aplicado a um sistema de
    segunda ordem (planta típica de processos industriais).
    Permite ajuste interativo dos ganhos Kp, Ki, Kd e
    visualização em tempo real da resposta ao degrau.

  Como usar:
    1. Execute: python pid_simulator.py
    2. Ajuste os valores de Kp, Ki, Kd nas caixas de texto
    3. Clique em "Simular" ou pressione Enter
    4. Analise o gráfico e as métricas de desempenho

  Conceitos abordados:
    - Controle PID (Proporcional, Integral, Derivativo)
    - Resposta ao degrau unitário
    - Métricas: overshoot, tempo de subida, tempo de acomodação
    - Discretização por método de Euler (integração numérica)
=============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# ─────────────────────────────────────────────
#  PLANTA DO SISTEMA (modelo matemático)
# ─────────────────────────────────────────────
class Planta:
    """
    Representa o sistema físico a ser controlado.

    Modelo: sistema de 2ª ordem com atraso.

    Função de transferência:
        G(s) = K / (tau1*s + 1)(tau2*s + 1)

    Parâmetros configuráveis via interface futuramente.
    """

    def __init__(self, K=1.0, tau1=1.0, tau2=0.5):
        self.K    = K       # Ganho estático
        self.tau1 = tau1    # Constante de tempo 1 (s)
        self.tau2 = tau2    # Constante de tempo 2 (s)
        self.reset()

    def reset(self):
        """Zera os estados internos da planta."""
        self.x1 = 0.0   # Estado 1
        self.x2 = 0.0   # Estado 2

    def passo(self, u, dt):
        """
        Avança a simulação em um passo de tempo dt (método de Euler).

        Equações de estado derivadas da FT de 2ª ordem:
          dx1/dt = x2
          dx2/dt = (K*u - x1 - (tau1+tau2)*x2) / (tau1*tau2)

        Parâmetros:
          u  : sinal de controle (saída do PID)
          dt : passo de tempo (s)

        Retorna:
          y  : saída da planta (variável controlada)
        """
        dx1 = self.x2
        dx2 = (self.K * u - self.x1 - (self.tau1 + self.tau2) * self.x2) / (self.tau1 * self.tau2)

        self.x1 += dx1 * dt
        self.x2 += dx2 * dt

        return self.x1   # Saída = primeiro estado

# ─────────────────────────────────────────────
#  CONTROLADOR PID
# ─────────────────────────────────────────────
class ControladorPID:
    """
    Implementação discreta do controlador PID.

    Lei de controle:
      u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt

    Onde:
      e(t) = referência - saída medida  (erro)
      Kp   = ganho proporcional
      Ki   = ganho integral
      Kd   = ganho derivativo
    """

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, u_min=-10.0, u_max=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min   # Saturação mínima do atuador
        self.u_max = u_max   # Saturação máxima do atuador
        self.reset()

    def reset(self):
        """Zera as variáveis de memória do controlador."""
        self.integral   = 0.0
        self.erro_prev  = 0.0

    def calcular(self, referencia, medicao, dt):
        """
        Calcula o sinal de controle u para um instante.

        Parâmetros:
          referencia : valor desejado (setpoint)
          medicao    : valor atual da saída
          dt         : passo de tempo (s)

        Retorna:
          u          : sinal de controle (saturado)
          componentes: dicionário com P, I, D separados
        """
        erro = referencia - medicao

        # Termo Proporcional
        P = self.Kp * erro

        # Termo Integral (método de Euler + anti-windup simples)
        self.integral += erro * dt
        I = self.Ki * self.integral

        # Termo Derivativo (derivada do erro)
        derivada = (erro - self.erro_prev) / dt if dt > 0 else 0.0
        D = self.Kd * derivada

        self.erro_prev = erro

        # Sinal de controle total com saturação
        u = P + I + D
        u_saturado = np.clip(u, self.u_min, self.u_max)

        # Anti-windup: corrige integral se houver saturação
        if u != u_saturado:
            self.integral -= erro * dt

        return u_saturado, {"P": P, "I": I, "D": D}

# ─────────────────────────────────────────────
#  FUNÇÕES DE SIMULAÇÃO E MÉTRICAS
# ─────────────────────────────────────────────
def simular(Kp, Ki, Kd, t_total=10.0, dt=0.01, referencia=1.0):
    """
    Executa a simulação completa do sistema em malha fechada.

    Parâmetros:
      Kp, Ki, Kd : ganhos do PID
      t_total    : duração total (s)
      dt         : passo de tempo (s)
      referencia : valor do degrau de entrada

    Retorna:
      t     : vetor de tempo
      y     : saída da planta
      u     : sinal de controle
      comp  : componentes P, I, D ao longo do tempo
    """
    planta = Planta()
    pid    = ControladorPID(Kp, Ki, Kd)

    n  = int(t_total / dt)
    t  = np.linspace(0, t_total, n)
    y  = np.zeros(n)
    u  = np.zeros(n)
    P_ = np.zeros(n)
    I_ = np.zeros(n)
    D_ = np.zeros(n)

    for i in range(1, n):
        controle, comp = pid.calcular(referencia, y[i-1], dt)
        y[i] = planta.passo(controle, dt)
        u[i] = controle
        P_[i] = comp["P"]
        I_[i] = comp["I"]
        D_[i] = comp["D"]

    return t, y, u, {"P": P_, "I": I_, "D": D_}


def calcular_metricas(t, y, referencia=1.0, tolerancia=0.02):
    """
    Calcula métricas clássicas de desempenho de controladores.

    Métricas calculadas:
      - Overshoot (%)       : ultrapassagem máxima
      - Tempo de subida (s) : 10% → 90% do valor final
      - Tempo de pico (s)   : instante do valor máximo
      - Tempo de acomodação : quando entra na banda ±2%
      - Erro em regime (%)  : erro estacionário

    Referência:
      Ogata, K. (2010). Modern Control Engineering, 5th Ed.
    """
    y_max   = np.max(y)
    y_final = y[-1]

    # Overshoot
    overshoot = max(0.0, (y_max - referencia) / referencia * 100)

    # Tempo de subida (10% a 90%)
    t_subida = None
    idx_10   = np.where(y >= 0.10 * referencia)[0]
    idx_90   = np.where(y >= 0.90 * referencia)[0]
    if len(idx_10) > 0 and len(idx_90) > 0:
        t_subida = t[idx_90[0]] - t[idx_10[0]]

    # Tempo de pico
    idx_pico = np.argmax(y)
    t_pico   = t[idx_pico]

    # Tempo de acomodação (banda ±tolerancia)
    banda_sup = referencia * (1 + tolerancia)
    banda_inf = referencia * (1 - tolerancia)
    t_acomodacao = None
    for i in range(len(y) - 1, -1, -1):
        if not (banda_inf <= y[i] <= banda_sup):
            if i + 1 < len(t):
                t_acomodacao = t[i + 1]
            break
    if t_acomodacao is None:
        t_acomodacao = t[0]

    # Erro em regime estacionário
    erro_regime = abs(referencia - y_final) / referencia * 100

    return {
        "overshoot":     round(overshoot, 2),
        "t_subida":      round(t_subida, 3) if t_subida else "—",
        "t_pico":        round(t_pico, 3),
        "t_acomodacao":  round(t_acomodacao, 3),
        "erro_regime":   round(erro_regime, 2),
    }

# ─────────────────────────────────────────────
#  INTERFACE GRÁFICA (tkinter + matplotlib)
# ─────────────────────────────────────────────
class AppPID:
    """
    Interface gráfica do Simulador PID.

    Layout:
      Esquerda  : painel de configuração (ganhos + métricas)
      Direita   : gráficos (resposta + sinal de controle + componentes PID)
    """

    COR_FUNDO      = "#0f1117"
    COR_PAINEL     = "#1a1d27"
    COR_BORDA      = "#2a2d3e"
    COR_DESTAQUE   = "#00d4ff"
    COR_TEXTO      = "#e8eaf0"
    COR_TEXTO_SUB  = "#8890aa"
    COR_SUCESSO    = "#00e676"
    COR_ALERTA     = "#ff9800"
    COR_ERRO_COR   = "#ff5252"
    FONTE_TITULO   = ("Consolas", 13, "bold")
    FONTE_LABEL    = ("Consolas", 10)
    FONTE_ENTRY    = ("Consolas", 12)
    FONTE_METRICA  = ("Consolas", 9)

    def __init__(self, root):
        self.root = root
        self.root.title("Simulador PID — Resposta ao Degrau")
        self.root.configure(bg=self.COR_FUNDO)
        self.root.resizable(True, True)

        self._construir_layout()
        self._simular_e_plotar()   # Simulação inicial com valores padrão

    # ── Layout principal ──────────────────────

    def _construir_layout(self):
        """Monta o layout principal: painel esquerdo + gráficos."""
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Painel de controle (esquerda)
        self.frame_controle = tk.Frame(
            self.root, bg=self.COR_PAINEL,
            width=280, padx=18, pady=18
        )
        self.frame_controle.grid(row=0, column=0, sticky="nsew")
        self.frame_controle.grid_propagate(False)

        # Área dos gráficos (direita)
        self.frame_graficos = tk.Frame(self.root, bg=self.COR_FUNDO)
        self.frame_graficos.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self._construir_painel_controle()
        self._construir_graficos()

    # ── Painel esquerdo ───────────────────────

    def _construir_painel_controle(self):
        f = self.frame_controle

        # Título
        tk.Label(f, text="⚙ SIMULADOR PID",
                 font=("Consolas", 14, "bold"),
                 fg=self.COR_DESTAQUE, bg=self.COR_PAINEL
                 ).pack(anchor="w", pady=(0, 4))

        tk.Label(f, text="Resposta ao Degrau Unitário",
                 font=("Consolas", 9),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL
                 ).pack(anchor="w", pady=(0, 16))

        self._separador(f)

        # ── Ganhos PID ──
        tk.Label(f, text="GANHOS DO CONTROLADOR",
                 font=("Consolas", 9, "bold"),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL
                 ).pack(anchor="w", pady=(12, 6))

        self.entry_Kp = self._campo_entrada(f, "Kp  (Proporcional)", "1.5")
        self.entry_Ki = self._campo_entrada(f, "Ki  (Integral)",     "0.8")
        self.entry_Kd = self._campo_entrada(f, "Kd  (Derivativo)",   "0.2")

        self._separador(f)

        # ── Parâmetros de simulação ──
        tk.Label(f, text="PARÂMETROS DA SIMULAÇÃO",
                 font=("Consolas", 9, "bold"),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL
                 ).pack(anchor="w", pady=(12, 6))

        self.entry_t = self._campo_entrada(f, "Tempo total (s)", "10")
        self.entry_ref = self._campo_entrada(f, "Referência (degrau)", "1.0")

        self._separador(f)

        # ── Botão simular ──
        self.btn_simular = tk.Button(
            f, text="▶  SIMULAR",
            font=("Consolas", 11, "bold"),
            fg=self.COR_FUNDO, bg=self.COR_DESTAQUE,
            activebackground="#00a8cc",
            relief="flat", cursor="hand2",
            command=self._simular_e_plotar,
            pady=8
        )
        self.btn_simular.pack(fill="x", pady=(16, 8))

        # Bind Enter nos campos
        for entry in [self.entry_Kp, self.entry_Ki, self.entry_Kd,
                      self.entry_t, self.entry_ref]:
            entry.bind("<Return>", lambda e: self._simular_e_plotar())

        # Botão reset
        tk.Button(
            f, text="↺  Resetar Padrões",
            font=("Consolas", 9),
            fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL,
            activebackground=self.COR_BORDA,
            relief="flat", cursor="hand2",
            command=self._resetar_valores
        ).pack(fill="x", pady=(0, 16))

        self._separador(f)

        # ── Métricas de desempenho ──
        tk.Label(f, text="MÉTRICAS DE DESEMPENHO",
                 font=("Consolas", 9, "bold"),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL
                 ).pack(anchor="w", pady=(12, 6))

        self.labels_metricas = {}
        metricas_def = [
            ("overshoot",    "Overshoot",      "%"),
            ("t_subida",     "Tempo subida",   "s"),
            ("t_pico",       "Tempo de pico",  "s"),
            ("t_acomodacao", "Acomodação (2%)", "s"),
            ("erro_regime",  "Erro regime",    "%"),
        ]
        for chave, nome, unidade in metricas_def:
            row = tk.Frame(f, bg=self.COR_PAINEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=nome, font=self.FONTE_METRICA,
                     fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL, width=16, anchor="w"
                     ).pack(side="left")
            lbl = tk.Label(row, text="—", font=("Consolas", 9, "bold"),
                           fg=self.COR_SUCESSO, bg=self.COR_PAINEL, anchor="e")
            lbl.pack(side="right")
            self.labels_metricas[chave] = (lbl, unidade)

        self._separador(f)

        # Rodapé
        tk.Label(f,
                 text="Planta: K/(τ₁s+1)(τ₂s+1)\nK=1  τ₁=1s  τ₂=0.5s",
                 font=("Consolas", 8),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL,
                 justify="left"
                 ).pack(anchor="w", pady=(12, 0))

    def _campo_entrada(self, parent, label, valor_padrao):
        """Cria um campo label + entry estilizado."""
        tk.Label(parent, text=label, font=("Consolas", 9),
                 fg=self.COR_TEXTO_SUB, bg=self.COR_PAINEL
                 ).pack(anchor="w")
        entry = tk.Entry(
            parent,
            font=self.FONTE_ENTRY,
            fg=self.COR_DESTAQUE,
            bg="#0f1117",
            insertbackground=self.COR_DESTAQUE,
            relief="flat",
            bd=6,
            width=18
        )
        entry.insert(0, valor_padrao)
        entry.pack(fill="x", pady=(2, 10))
        return entry

    def _separador(self, parent):
        tk.Frame(parent, bg=self.COR_BORDA, height=1).pack(fill="x", pady=4)

    # ── Área de gráficos ──────────────────────

    def _construir_graficos(self):
        """Cria a figura matplotlib com 3 subplots."""
        self.fig = plt.Figure(figsize=(9, 7), dpi=100, facecolor=self.COR_FUNDO)

        gs = gridspec.GridSpec(3, 1, figure=self.fig,
                               hspace=0.45,
                               top=0.94, bottom=0.08,
                               left=0.08, right=0.97)

        estilo = dict(facecolor="#12151f", labelcolor=self.COR_TEXTO_SUB,
                      tick_params={"colors": self.COR_TEXTO_SUB})

        self.ax_resp = self.fig.add_subplot(gs[0])
        self.ax_ctrl = self.fig.add_subplot(gs[1])
        self.ax_comp = self.fig.add_subplot(gs[2])

        for ax in [self.ax_resp, self.ax_ctrl, self.ax_comp]:
            ax.set_facecolor("#12151f")
            ax.tick_params(colors=self.COR_TEXTO_SUB, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.COR_BORDA)
            ax.grid(True, color=self.COR_BORDA, linewidth=0.5, linestyle="--")
            ax.yaxis.label.set_color(self.COR_TEXTO_SUB)
            ax.xaxis.label.set_color(self.COR_TEXTO_SUB)
            ax.title.set_color(self.COR_TEXTO)

        self.ax_resp.set_title("Resposta ao Degrau", fontsize=10, fontweight="bold", pad=8)
        self.ax_ctrl.set_title("Sinal de Controle u(t)", fontsize=10, fontweight="bold", pad=8)
        self.ax_comp.set_title("Componentes PID", fontsize=10, fontweight="bold", pad=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graficos)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Lógica de simulação ───────────────────

    def _ler_parametros(self):
        """Lê e valida os parâmetros digitados pelo usuário."""
        try:
            Kp  = float(self.entry_Kp.get())
            Ki  = float(self.entry_Ki.get())
            Kd  = float(self.entry_Kd.get())
            t   = float(self.entry_t.get())
            ref = float(self.entry_ref.get())
            if t <= 0:
                raise ValueError("Tempo total deve ser positivo.")
            if ref <= 0:
                raise ValueError("Referência deve ser positiva.")
            return Kp, Ki, Kd, t, ref
        except ValueError as e:
            messagebox.showerror("Entrada inválida",
                                 f"Verifique os valores inseridos.\n\nDetalhe: {e}")
            return None

    def _simular_e_plotar(self):
        """Executa a simulação e atualiza todos os gráficos e métricas."""
        params = self._ler_parametros()
        if params is None:
            return

        Kp, Ki, Kd, t_total, ref = params

        # Simulação
        t, y, u, comp = simular(Kp, Ki, Kd, t_total=t_total, referencia=ref)
        metricas = calcular_metricas(t, y, referencia=ref)

        # ── Gráfico 1: Resposta ──
        self.ax_resp.cla()
        self.ax_resp.set_facecolor("#12151f")
        self.ax_resp.plot(t, y, color=self.COR_DESTAQUE, linewidth=2.0, label="Saída y(t)")
        self.ax_resp.axhline(ref, color=self.COR_SUCESSO, linewidth=1.2,
                             linestyle="--", label=f"Referência = {ref}")
        self.ax_resp.axhline(ref * 1.02, color="#444", linewidth=0.8, linestyle=":")
        self.ax_resp.axhline(ref * 0.98, color="#444", linewidth=0.8, linestyle=":")
        self.ax_resp.fill_between(t, ref * 0.98, ref * 1.02, alpha=0.06, color=self.COR_SUCESSO)
        self.ax_resp.set_ylabel("Amplitude", fontsize=9)
        self.ax_resp.set_title("Resposta ao Degrau", fontsize=10,
                               fontweight="bold", color=self.COR_TEXTO, pad=8)
        self.ax_resp.legend(fontsize=8, facecolor="#1a1d27",
                            labelcolor=self.COR_TEXTO, edgecolor=self.COR_BORDA)
        self.ax_resp.tick_params(colors=self.COR_TEXTO_SUB, labelsize=8)
        for spine in self.ax_resp.spines.values():
            spine.set_edgecolor(self.COR_BORDA)
        self.ax_resp.grid(True, color=self.COR_BORDA, linewidth=0.5, linestyle="--")

        # Anotação do overshoot
        if metricas["overshoot"] > 0.5:
            idx_pico = np.argmax(y)
            self.ax_resp.annotate(
                f"  OS={metricas['overshoot']}%",
                xy=(t[idx_pico], y[idx_pico]),
                fontsize=8, color=self.COR_ALERTA,
                arrowprops=dict(arrowstyle="->", color=self.COR_ALERTA, lw=1.0),
                xytext=(t[idx_pico] + t_total * 0.05, y[idx_pico])
            )

        # ── Gráfico 2: Sinal de controle ──
        self.ax_ctrl.cla()
        self.ax_ctrl.set_facecolor("#12151f")
        self.ax_ctrl.plot(t, u, color=self.COR_ALERTA, linewidth=1.8, label="u(t)")
        self.ax_ctrl.set_ylabel("u(t)", fontsize=9)
        self.ax_ctrl.set_title("Sinal de Controle u(t)", fontsize=10,
                               fontweight="bold", color=self.COR_TEXTO, pad=8)
        self.ax_ctrl.legend(fontsize=8, facecolor="#1a1d27",
                            labelcolor=self.COR_TEXTO, edgecolor=self.COR_BORDA)
        self.ax_ctrl.tick_params(colors=self.COR_TEXTO_SUB, labelsize=8)
        for spine in self.ax_ctrl.spines.values():
            spine.set_edgecolor(self.COR_BORDA)
        self.ax_ctrl.grid(True, color=self.COR_BORDA, linewidth=0.5, linestyle="--")

        # ── Gráfico 3: Componentes P, I, D ──
        self.ax_comp.cla()
        self.ax_comp.set_facecolor("#12151f")
        self.ax_comp.plot(t, comp["P"], color="#ff6b6b", linewidth=1.5, label="P")
        self.ax_comp.plot(t, comp["I"], color="#a29bfe", linewidth=1.5, label="I")
        self.ax_comp.plot(t, comp["D"], color="#55efc4", linewidth=1.5, label="D")
        self.ax_comp.set_xlabel("Tempo (s)", fontsize=9)
        self.ax_comp.set_ylabel("Amplitude", fontsize=9)
        self.ax_comp.set_title("Componentes PID", fontsize=10,
                               fontweight="bold", color=self.COR_TEXTO, pad=8)
        self.ax_comp.legend(fontsize=8, facecolor="#1a1d27",
                            labelcolor=self.COR_TEXTO, edgecolor=self.COR_BORDA,
                            ncol=3)
        self.ax_comp.tick_params(colors=self.COR_TEXTO_SUB, labelsize=8)
        for spine in self.ax_comp.spines.values():
            spine.set_edgecolor(self.COR_BORDA)
        self.ax_comp.grid(True, color=self.COR_BORDA, linewidth=0.5, linestyle="--")

        self.canvas.draw()

        # ── Atualizar métricas ──
        cores = {
            "overshoot":    (self.COR_ALERTA if metricas["overshoot"] > 10 else self.COR_SUCESSO),
            "t_subida":     self.COR_DESTAQUE,
            "t_pico":       self.COR_DESTAQUE,
            "t_acomodacao": self.COR_DESTAQUE,
            "erro_regime":  (self.COR_ERRO_COR if metricas["erro_regime"] > 5 else self.COR_SUCESSO),
        }
        for chave, (lbl, unidade) in self.labels_metricas.items():
            valor = metricas[chave]
            texto = f"{valor} {unidade}" if valor != "—" else "—"
            lbl.config(text=texto, fg=cores.get(chave, self.COR_SUCESSO))

    def _resetar_valores(self):
        """Restaura os valores padrão dos campos."""
        for entry, val in [
            (self.entry_Kp,  "1.5"),
            (self.entry_Ki,  "0.8"),
            (self.entry_Kd,  "0.2"),
            (self.entry_t,   "10"),
            (self.entry_ref, "1.0"),
        ]:
            entry.delete(0, tk.END)
            entry.insert(0, val)
        self._simular_e_plotar()

# ─────────────────────────────────────────────
#  PONTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x720")
    app = AppPID(root)
    root.mainloop()
