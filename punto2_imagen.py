"""
Quiz 2 - Punto 2: Algoritmo Genético
Generación de Imagen con figuras geométricas
Interfaz gráfica con Tkinter + Matplotlib
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw
import threading

# ============================================================
#  PARÁMETROS GLOBALES
# ============================================================
TAM           = 64
N_FIGURAS     = 30
TAM_POBLACION = 30
PROB_MUTACION = 0.05
PROB_CRUCE    = 0.8

# ============================================================
#  IMAGEN OBJETIVO
#  Pon aqui la ruta de tu imagen. Ejemplos:
#    RUTA_IMAGEN = r"C:\Users\USUARIO\Pictures\foto.jpg"
#    RUTA_IMAGEN = r"C:\Users\USUARIO\Desktop\logo.png"
#  Si la ruta es None o el archivo no existe, se usa la
#  bandera de Colombia como imagen de ejemplo.
# ============================================================
RUTA_IMAGEN = r"C:\Users\USUARIO\Downloads\Flag_UK.png.jpg"  # <-- CAMBIA ESTO por tu ruta

def crear_objetivo(tam):
    if RUTA_IMAGEN:
        try:
            img = Image.open(RUTA_IMAGEN).convert("RGB")
            img = img.resize((tam, tam), Image.LANCZOS)
            return img
        except Exception as e:
            print(f"[Aviso] No se pudo cargar '{RUTA_IMAGEN}': {e}")
            print("[Aviso] Usando imagen de ejemplo (bandera de Colombia).")

    # Fallback: bandera de Colombia
    img = Image.new("RGB", (tam, tam))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0,        0, tam, tam//2],   fill=(255, 210, 0))
    draw.rectangle([0,   tam//2, tam, tam*3//4], fill=(0,   56,  147))
    draw.rectangle([0, tam*3//4, tam, tam],       fill=(206, 17,  38))
    return img

# ============================================================
#  GENÉTICA
# ============================================================
def gen_aleatorio(tam):
    return [
        random.randint(0, tam),
        random.randint(0, tam),
        random.randint(5, tam // 2),
        random.randint(5, tam // 2),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(30, 200),
        random.randint(0, 1),
    ]

def individuo_aleatorio(n_fig, tam):
    return [gen_aleatorio(tam) for _ in range(n_fig)]

def renderizar(individuo, tam):
    canvas = Image.new("RGB", (tam, tam), (255, 255, 255))
    overlay = Image.new("RGBA", (tam, tam), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for g in individuo:
        x, y, w, h, r, gr, b, a, tipo = g
        bbox = [x, y, x + w, y + h]
        color = (r, gr, b, a)
        if tipo == 0:
            draw.rectangle(bbox, fill=color)
        else:
            draw.ellipse(bbox, fill=color)
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))
    return canvas

def fitness(individuo, arr_obj, tam):
    arr = np.array(renderizar(individuo, tam), dtype=np.float32)
    return -float(np.mean((arr - arr_obj) ** 2))

def cruce(p1, p2):
    if random.random() < PROB_CRUCE:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return [g[:] for g in p1], [g[:] for g in p2]

def mutar(ind, tam):
    result = []
    for g in ind:
        if random.random() < PROB_MUTACION:
            if random.random() < 0.3:
                result.append(gen_aleatorio(tam))
            else:
                g = g[:]
                idx = random.randint(0, 7)
                d = random.randint(-30, 30)
                if idx < 2:
                    g[idx] = max(0, min(tam, g[idx] + d))
                elif idx < 4:
                    g[idx] = max(5, min(tam // 2, g[idx] + d))
                else:
                    g[idx] = max(0, min(255, g[idx] + d))
                result.append(g)
        else:
            result.append(g[:])
    return result

def seleccion_torneo(pob, fits, k=3):
    comp = random.sample(range(len(pob)), k)
    win = max(comp, key=lambda i: fits[i])
    return [g[:] for g in pob[win]]


# ============================================================
#  INTERFAZ GRÁFICA
# ============================================================
class AppImagen(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎨 AG — Generación de Imagen | Quiz 2 IA")
        self.geometry("1150x680")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self._corriendo = False
        self._build_ui()

    # ----------------------------------------------------------
    def _build_ui(self):
        # ---- Panel izquierdo ----
        left = tk.Frame(self, bg="#2a2a3e", width=270)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="⚙️ Parámetros", font=("Helvetica", 13, "bold"),
                 bg="#2a2a3e", fg="#cdd6f4").pack(pady=(15, 5))

        self.vars = {}
        params = [
            ("Generaciones",  "generaciones",  300,  50,  800, 50),
            ("Población",     "poblacion",      30,  10,   80,  5),
            ("# Figuras",     "figuras",        30,   5,   70,  5),
            ("P. Mutación",   "mutacion",     0.05, 0.01,  0.3, 0.01),
            ("Tamaño imagen", "tam",            64,  32,   96, 16),
        ]
        for label, key, default, mn, mx, step in params:
            frm = tk.Frame(left, bg="#2a2a3e")
            frm.pack(fill=tk.X, padx=15, pady=4)
            tk.Label(frm, text=label, bg="#2a2a3e", fg="#a6adc8",
                     font=("Helvetica", 9)).pack(anchor=tk.W)
            var = tk.DoubleVar(value=default)
            self.vars[key] = var
            tk.Scale(frm, variable=var, from_=mn, to=mx,
                     resolution=step, orient=tk.HORIZONTAL,
                     bg="#2a2a3e", fg="#cdd6f4", troughcolor="#45475a",
                     highlightthickness=0, length=220).pack()

        self.btn = tk.Button(left, text="▶  Iniciar AG",
                             font=("Helvetica", 11, "bold"),
                             bg="#89b4fa", fg="#1e1e2e",
                             activebackground="#74c7ec",
                             relief=tk.FLAT, cursor="hand2",
                             command=self._run_thread)
        self.btn.pack(pady=12, padx=15, fill=tk.X)

        self.progress = ttk.Progressbar(left, length=220, mode="determinate")
        self.progress.pack(padx=15, pady=4)

        self.lbl_estado = tk.Label(left, text="", bg="#2a2a3e", fg="#a6e3a1",
                                   font=("Courier", 9))
        self.lbl_estado.pack(pady=4)

        # Imagen objetivo pequeña
        tk.Label(left, text="🎯 Imagen Objetivo", bg="#2a2a3e",
                 fg="#cdd6f4", font=("Helvetica", 9, "bold")).pack(pady=(10, 2))
        self.lbl_obj = tk.Label(left, bg="#2a2a3e")
        self.lbl_obj.pack()
        self._mostrar_objetivo(64)

        # ---- Panel derecho ----
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Fila superior: evolución de snapshots
        self.fig_snaps, self.axes_snaps = plt.subplots(2, 4, figsize=(10, 4))
        self.fig_snaps.patch.set_facecolor("#1e1e2e")
        for ax in self.axes_snaps.flat:
            ax.set_facecolor("#313244")
            ax.axis("off")
        self.canvas_snaps = FigureCanvasTkAgg(self.fig_snaps, master=right)
        self.canvas_snaps.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Fila inferior: curva MSE
        self.fig_mse, self.ax_mse = plt.subplots(figsize=(10, 2))
        self.fig_mse.patch.set_facecolor("#1e1e2e")
        self.ax_mse.set_facecolor("#313244")
        self.ax_mse.set_title("Convergencia MSE", color="#cdd6f4", fontsize=9)
        self.ax_mse.set_xlabel("Generación", color="#a6adc8", fontsize=8)
        self.ax_mse.set_ylabel("MSE", color="#a6adc8", fontsize=8)
        self.ax_mse.tick_params(colors="#a6adc8")
        self.fig_mse.tight_layout(pad=1.5)
        self.canvas_mse = FigureCanvasTkAgg(self.fig_mse, master=right)
        self.canvas_mse.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------
    def _mostrar_objetivo(self, tam):
        img = crear_objetivo(tam)
        img_tk = img.resize((80, 80), Image.NEAREST)
        from PIL import ImageTk
        self._img_obj_tk = ImageTk.PhotoImage(img_tk)
        self.lbl_obj.config(image=self._img_obj_tk)

    # ----------------------------------------------------------
    def _run_thread(self):
        if self._corriendo:
            return
        self._corriendo = True
        self.btn.config(state=tk.DISABLED, text="⏳ Ejecutando...")
        t = threading.Thread(target=self._ejecutar, daemon=True)
        t.start()

    # ----------------------------------------------------------
    def _ejecutar(self):
        global PROB_MUTACION
        n_gen   = int(self.vars["generaciones"].get())
        tam_pob = int(self.vars["poblacion"].get())
        n_fig   = int(self.vars["figuras"].get())
        PROB_MUTACION = self.vars["mutacion"].get()
        tam     = int(self.vars["tam"].get())

        self.after(0, lambda: self._mostrar_objetivo(tam))

        img_obj  = crear_objetivo(tam)
        arr_obj  = np.array(img_obj, dtype=np.float32)
        pob      = [individuo_aleatorio(n_fig, tam) for _ in range(tam_pob)]
        hist_mse = []
        snapshots = []
        snap_gens = sorted(set([0, n_gen//6, n_gen//3, n_gen//2,
                                 2*n_gen//3, 5*n_gen//6, n_gen-1]))
        mejor_global = None
        mejor_fit    = -np.inf

        for gen in range(n_gen):
            fits = [fitness(ind, arr_obj, tam) for ind in pob]
            idx  = int(np.argmax(fits))
            fm   = fits[idx]
            hist_mse.append(-fm)

            if fm > mejor_fit:
                mejor_fit    = fm
                mejor_global = [g[:] for g in pob[idx]]

            if gen in snap_gens:
                snapshots.append((gen, renderizar(pob[idx], tam), -fm))

            # Actualizar UI cada 20 generaciones
            if gen % 20 == 0 or gen == n_gen - 1:
                pct = int((gen + 1) / n_gen * 100)
                mse_actual = -fm
                self.after(0, lambda p=pct, m=mse_actual, g=gen:
                           self._update_progreso(p, m, g))

            # Nueva generación
            elite  = [[g[:] for g in pob[idx]]]
            nueva  = elite[:]
            while len(nueva) < tam_pob:
                h1, h2 = cruce(
                    seleccion_torneo(pob, fits),
                    seleccion_torneo(pob, fits)
                )
                nueva += [mutar(h1, tam), mutar(h2, tam)]
            pob = nueva[:tam_pob]

        self.after(0, lambda: self._mostrar_resultados(
            snapshots, hist_mse, mejor_global, img_obj, tam))

    # ----------------------------------------------------------
    def _update_progreso(self, pct, mse, gen):
        self.progress["value"] = pct
        self.lbl_estado.config(text=f"Gen {gen} | MSE={mse:.1f}")
        self.update_idletasks()

    # ----------------------------------------------------------
    def _mostrar_resultados(self, snapshots, hist_mse, mejor, img_obj, tam):
        # Snapshots
        for i, ax in enumerate(self.axes_snaps.flat):
            ax.clear()
            ax.set_facecolor("#313244")
            ax.axis("off")

        for i, (gen, img_snap, mse) in enumerate(snapshots[:7]):
            ax = self.axes_snaps.flat[i]
            ax.imshow(np.array(img_snap))
            ax.set_title(f"Gen {gen}\nMSE={mse:.0f}",
                         color="#cdd6f4", fontsize=7)
            ax.axis("off")

        # Última celda: objetivo
        ax_last = self.axes_snaps.flat[7]
        ax_last.imshow(np.array(img_obj))
        ax_last.set_title("🎯 Objetivo", color="#a6e3a1", fontsize=8)
        ax_last.axis("off")

        self.fig_snaps.suptitle("Evolución del AG — Generación de Imagen",
                                color="#cdd6f4", fontsize=10)
        self.fig_snaps.tight_layout(pad=1.5)
        self.canvas_snaps.draw()

        # Curva MSE
        self.ax_mse.clear()
        self.ax_mse.set_facecolor("#313244")
        self.ax_mse.plot(hist_mse, color="#f38ba8", linewidth=1.5)
        self.ax_mse.set_title("Convergencia MSE (menor = mejor)",
                              color="#cdd6f4", fontsize=9)
        self.ax_mse.set_xlabel("Generación", color="#a6adc8", fontsize=8)
        self.ax_mse.set_ylabel("MSE", color="#a6adc8", fontsize=8)
        self.ax_mse.tick_params(colors="#a6adc8")
        self.ax_mse.grid(True, alpha=0.2)
        self.fig_mse.tight_layout(pad=1.5)
        self.canvas_mse.draw()

        mse_final = hist_mse[-1] if hist_mse else 0
        self.lbl_estado.config(
            text=f"✅ Listo | MSE final={mse_final:.1f}")
        self.btn.config(state=tk.NORMAL, text="▶  Iniciar AG")
        self.progress["value"] = 100
        self._corriendo = False


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    app = AppImagen()
    app.mainloop()