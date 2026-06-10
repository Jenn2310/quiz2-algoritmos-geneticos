"""
Quiz 2 - Punto 1: Algoritmo Genético
Problema de la Mochila (0/1 Knapsack)
Interfaz gráfica con Tkinter + Matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# ============================================================
#  DATOS DEL PROBLEMA
# ============================================================
ITEMS = [
    {'nombre': 'Laptop',               'peso': 3.0, 'valor': 150},
    {'nombre': 'Cámara',               'peso': 1.0, 'valor': 80},
    {'nombre': 'Libros',               'peso': 2.5, 'valor': 40},
    {'nombre': 'Ropa',                 'peso': 2.0, 'valor': 30},
    {'nombre': 'Tablet',               'peso': 1.5, 'valor': 90},
    {'nombre': 'Auriculares',          'peso': 0.5, 'valor': 50},
    {'nombre': 'Cargadores',           'peso': 0.5, 'valor': 20},
    {'nombre': 'Botella agua',         'peso': 0.8, 'valor': 10},
    {'nombre': 'Snacks',               'peso': 1.2, 'valor': 15},
    {'nombre': 'Kit primeros auxilios','peso': 0.7, 'valor': 35},
    {'nombre': 'Linterna',             'peso': 0.4, 'valor': 25},
    {'nombre': 'Navaja suiza',         'peso': 0.3, 'valor': 45},
]
PESOS   = np.array([i['peso']  for i in ITEMS])
VALORES = np.array([i['valor'] for i in ITEMS])
NOMBRES = [i['nombre'] for i in ITEMS]
N_ITEMS = len(ITEMS)

# ============================================================
#  ALGORITMO GENÉTICO
# ============================================================
def calcular_fitness(cromosoma, capacidad):
    if np.dot(cromosoma, PESOS) > capacidad:
        return 0
    return float(np.dot(cromosoma, VALORES))

def crear_poblacion(tam, n):
    return [np.random.randint(0, 2, n) for _ in range(tam)]

def seleccion_torneo(poblacion, fitnesses, k=3):
    competidores = random.sample(range(len(poblacion)), k)
    ganador = max(competidores, key=lambda i: fitnesses[i])
    return poblacion[ganador].copy()

def cruce(p1, p2, prob):
    if random.random() < prob:
        punto = random.randint(1, len(p1) - 1)
        return np.concatenate([p1[:punto], p2[punto:]]), np.concatenate([p2[:punto], p1[punto:]])
    return p1.copy(), p2.copy()

def mutar(cromosoma, prob):
    for i in range(len(cromosoma)):
        if random.random() < prob:
            cromosoma[i] = 1 - cromosoma[i]
    return cromosoma

def algoritmo_genetico(tam_pob, n_gen, prob_cruce, prob_mut, capacidad, callback=None):
    random.seed(42)
    np.random.seed(42)
    poblacion = crear_poblacion(tam_pob, N_ITEMS)
    hist_mejor, hist_prom = [], []
    mejor_global, mejor_fit = None, 0

    for gen in range(n_gen):
        fitnesses = [calcular_fitness(c, capacidad) for c in poblacion]
        idx = int(np.argmax(fitnesses))
        fm = fitnesses[idx]
        hist_mejor.append(fm)
        hist_prom.append(float(np.mean(fitnesses)))

        if fm > mejor_fit:
            mejor_fit = fm
            mejor_global = poblacion[idx].copy()

        if callback:
            callback(gen + 1, n_gen, fm)

        # Elitismo + nueva generación
        elite = [poblacion[idx].copy()]
        nueva = elite[:]
        while len(nueva) < tam_pob:
            h1, h2 = cruce(
                seleccion_torneo(poblacion, fitnesses),
                seleccion_torneo(poblacion, fitnesses),
                prob_cruce
            )
            nueva += [mutar(h1, prob_mut), mutar(h2, prob_mut)]
        poblacion = nueva[:tam_pob]

    return mejor_global, mejor_fit, hist_mejor, hist_prom


# ============================================================
#  INTERFAZ GRÁFICA
# ============================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧬 AG — Problema de la Mochila | Quiz 2 IA")
        self.geometry("1100x700")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self._build_ui()

    # ----------------------------------------------------------
    def _build_ui(self):
        # ---- Panel izquierdo: controles ----
        left = tk.Frame(self, bg="#2a2a3e", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="⚙️ Parámetros", font=("Helvetica", 13, "bold"),
                 bg="#2a2a3e", fg="#cdd6f4").pack(pady=(15, 5))

        self.vars = {}
        params = [
            ("Población",    "poblacion",    100, 20,  500, 1),
            ("Generaciones", "generaciones", 300, 50, 1000, 50),
            ("P. Cruce",     "cruce",       0.85, 0.1,  1.0, 0.05),
            ("P. Mutación",  "mutacion",    0.03, 0.001, 0.3, 0.005),
            ("Capacidad (kg)","capacidad",   8.0,  2.0, 20.0, 0.5),
        ]
        for label, key, default, mn, mx, step in params:
            frm = tk.Frame(left, bg="#2a2a3e")
            frm.pack(fill=tk.X, padx=15, pady=4)
            tk.Label(frm, text=label, bg="#2a2a3e", fg="#a6adc8",
                     font=("Helvetica", 9)).pack(anchor=tk.W)
            var = tk.DoubleVar(value=default)
            self.vars[key] = var
            scale = tk.Scale(frm, variable=var, from_=mn, to=mx,
                             resolution=step, orient=tk.HORIZONTAL,
                             bg="#2a2a3e", fg="#cdd6f4", troughcolor="#45475a",
                             highlightthickness=0, length=230)
            scale.pack()

        # Botón ejecutar
        self.btn = tk.Button(left, text="▶  Ejecutar AG", font=("Helvetica", 11, "bold"),
                             bg="#a6e3a1", fg="#1e1e2e", activebackground="#94e2d5",
                             relief=tk.FLAT, cursor="hand2", command=self._run_thread)
        self.btn.pack(pady=15, padx=15, fill=tk.X)

        # Barra de progreso
        self.progress = ttk.Progressbar(left, length=230, mode="determinate")
        self.progress.pack(padx=15, pady=5)

        # Resultado textual
        tk.Label(left, text="📦 Resultado", font=("Helvetica", 11, "bold"),
                 bg="#2a2a3e", fg="#cdd6f4").pack(pady=(10, 2))
        self.result_text = tk.Text(left, height=14, bg="#313244", fg="#cdd6f4",
                                   font=("Courier", 8), relief=tk.FLAT, state=tk.DISABLED)
        self.result_text.pack(padx=10, fill=tk.X)

        # ---- Panel derecho: gráficas ----
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.axes = plt.subplots(1, 3, figsize=(10, 4))
        self.fig.patch.set_facecolor("#1e1e2e")
        for ax in self.axes:
            ax.set_facecolor("#313244")
            ax.tick_params(colors="#cdd6f4")
            for spine in ax.spines.values():
                spine.set_edgecolor("#45475a")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(pad=2)
        self.canvas.draw()

    # ----------------------------------------------------------
    def _update_progress(self, gen, total, fitness):
        pct = int(gen / total * 100)
        self.progress["value"] = pct
        self.update_idletasks()

    # ----------------------------------------------------------
    def _run_thread(self):
        self.btn.config(state=tk.DISABLED, text="⏳ Ejecutando...")
        t = threading.Thread(target=self._ejecutar, daemon=True)
        t.start()

    # ----------------------------------------------------------
    def _ejecutar(self):
        tam_pob  = int(self.vars["poblacion"].get())
        n_gen    = int(self.vars["generaciones"].get())
        p_cruce  = self.vars["cruce"].get()
        p_mut    = self.vars["mutacion"].get()
        cap      = self.vars["capacidad"].get()

        mejor, valor, hist_m, hist_p = algoritmo_genetico(
            tam_pob, n_gen, p_cruce, p_mut, cap,
            callback=self._update_progress
        )

        self.after(0, lambda: self._mostrar_resultados(mejor, valor, hist_m, hist_p, cap))

    # ----------------------------------------------------------
    def _mostrar_resultados(self, mejor, valor, hist_m, hist_p, cap):
        # Texto resultado
        peso_final = float(np.dot(mejor, PESOS))
        lines = [f"💰 Valor total : {valor:.0f}\n",
                 f"⚖️  Peso usado  : {peso_final:.2f}/{cap:.1f} kg\n",
                 "─" * 28 + "\n",
                 "Ítems incluidos:\n"]
        for i, inc in enumerate(mejor):
            if inc:
                lines.append(f"  ✔ {NOMBRES[i]}\n")

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "".join(lines))
        self.result_text.config(state=tk.DISABLED)

        # ---- Gráfica 1: Convergencia ----
        ax0 = self.axes[0]
        ax0.clear()
        ax0.set_facecolor("#313244")
        gens = range(len(hist_m))
        ax0.plot(gens, hist_m, color="#a6e3a1", linewidth=2, label="Mejor")
        ax0.plot(gens, hist_p, color="#89b4fa", linewidth=1.5,
                 linestyle="--", label="Promedio")
        ax0.axhline(y=valor, color="#f38ba8", linestyle=":", linewidth=1)
        ax0.set_title("Evolución Fitness", color="#cdd6f4", fontsize=9)
        ax0.set_xlabel("Generación", color="#a6adc8", fontsize=8)
        ax0.set_ylabel("Fitness", color="#a6adc8", fontsize=8)
        ax0.tick_params(colors="#a6adc8")
        ax0.legend(fontsize=7, facecolor="#313244", labelcolor="#cdd6f4")
        ax0.grid(True, alpha=0.2)

        # ---- Gráfica 2: Ítems ----
        ax1 = self.axes[1]
        ax1.clear()
        ax1.set_facecolor("#313244")
        colores = ["#a6e3a1" if g else "#f38ba8" for g in mejor]
        ax1.barh(NOMBRES, VALORES, color=colores, edgecolor="#1e1e2e")
        ax1.set_title("Ítems Seleccionados", color="#cdd6f4", fontsize=9)
        ax1.set_xlabel("Valor", color="#a6adc8", fontsize=8)
        ax1.tick_params(colors="#a6adc8", labelsize=7)
        verde = mpatches.Patch(color="#a6e3a1", label="Incluido")
        rojo  = mpatches.Patch(color="#f38ba8", label="Excluido")
        ax1.legend(handles=[verde, rojo], fontsize=7,
                   facecolor="#313244", labelcolor="#cdd6f4")
        ax1.grid(True, alpha=0.2, axis="x")

        # ---- Gráfica 3: Pie peso ----
        ax2 = self.axes[2]
        ax2.clear()
        ax2.set_facecolor("#313244")
        pesos_inc  = [PESOS[i] for i, g in enumerate(mejor) if g]
        nombres_inc = [NOMBRES[i] for i, g in enumerate(mejor) if g]
        libre = cap - sum(pesos_inc)
        colores_pie = plt.cm.Pastel1.colors[:len(pesos_inc)] + ((0.9, 0.9, 0.9),)
        ax2.pie(
            pesos_inc + [libre],
            labels=nombres_inc + ["Libre"],
            autopct="%1.1f%%",
            startangle=90,
            colors=list(colores_pie),
            textprops={"color": "#cdd6f4", "fontsize": 7}
        )
        ax2.set_title(f"Peso (cap={cap:.1f}kg)", color="#cdd6f4", fontsize=9)

        self.fig.tight_layout(pad=2)
        self.canvas.draw()

        self.btn.config(state=tk.NORMAL, text="▶  Ejecutar AG")
        self.progress["value"] = 100


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
