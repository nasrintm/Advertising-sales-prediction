"""
Advertising Sales Predictor - desktop application (Tkinter)

Run:  python desktop_app.py
Needs model.json (created by train_model.py), the dataset CSV and the plots/ folder next to this file.

Tabs:  Predict | Dataset | Model results | Graphs
"""
import csv
import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from predictor import InputError, SalesPredictor, resource_dir

try:  # Pillow is installed together with matplotlib; used only to resize the plot images
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False

DATA_FILE = "Advertising_Budget_and_Sales.csv"
BG, CARD, ACCENT, INK, MUTED = "#f3f5f8", "#ffffff", "#2a78d6", "#1f2328", "#6b7280"
ORANGE, GOOD = "#eb6834", "#1a7f4b"

PLOT_TITLES = {
    "01_distributions": "1. Distributions of budgets and sales",
    "02_correlation_heatmap": "2. Correlation heatmap",
    "03_features_vs_sales": "3. Each channel vs sales",
    "04_actual_vs_predicted": "4. Actual vs predicted (test set)",
    "05_residuals": "5. Residual analysis",
    "06_model_comparison": "6. Model comparison",
    "07_tv_radio_interaction": "7. TV x Radio interaction",
}


def read_csv_table(path):
    """Return (header, rows) from the CSV, dropping the unnamed index column."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    keep = [i for i, h in enumerate(header) if h.strip() and not h.startswith("Unnamed")]
    return [header[i] for i in keep], [[r[i] for i in keep] for r in body if r]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advertising Sales Predictor - Linear Regression")
        self.geometry("1060x740")
        self.minsize(940, 660)
        self.configure(bg=BG)
        self.base = resource_dir()

        try:
            self.pred = SalesPredictor(self.base / "model.json")
        except Exception as exc:
            messagebox.showerror(
                "Model not found",
                f"Could not load model.json:\n{exc}\n\nRun  python train_model.py  first, "
                "then start the app again.")
            self.destroy()
            sys.exit(1)

        self._photo = None          # keep a reference so Tk does not garbage-collect the image
        self._syncing = False       # True while code (not the user) moves a slider
        self._setup_style()

        header = tk.Frame(self, bg=ACCENT)
        header.pack(fill="x")
        tk.Label(header, text="Advertising Sales Predictor", bg=ACCENT, fg="white",
                 font=("Segoe UI", 17, "bold")).pack(side="left", padx=18, pady=10)
        tk.Label(header, text="Linear regression on the Advertising Budget and Sales dataset",
                 bg=ACCENT, fg="#dbe8fb", font=("Segoe UI", 10)).pack(side="left", pady=10)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)
        self.tab_predict = ttk.Frame(self.tabs, padding=12)
        self.tab_data = ttk.Frame(self.tabs, padding=12)
        self.tab_model = ttk.Frame(self.tabs, padding=12)
        self.tab_plots = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.tab_predict, text="  Predict  ")
        self.tabs.add(self.tab_data, text="  Dataset  ")
        self.tabs.add(self.tab_model, text="  Model results  ")
        self.tabs.add(self.tab_plots, text="  Graphs  ")

        self._build_predict_tab(self.tab_predict)
        self._build_data_tab(self.tab_data)
        self._build_model_tab(self.tab_model)
        self._build_plots_tab(self.tab_plots)
        self.predict()  # show a first prediction for the average budgets

    # ------------------------------------------------------------------ style
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 10, "bold"))
        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=BG)
        style.configure("TLabelframe.Label", background=BG, foreground=INK, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Warn.TLabel", foreground=ORANGE, font=("Segoe UI", 9, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    # ------------------------------------------------------------------ Predict tab
    def _build_predict_tab(self, parent):
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(parent, text="Advertising budgets", padding=14)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        self.vars, self.scales = {}, {}
        for r, feat in enumerate(self.pred.features):
            lo, hi = self.pred.ranges[feat]
            ttk.Label(left, text=self.pred.labels[feat]).grid(row=3 * r, column=0, sticky="w", columnspan=2)
            var = tk.StringVar(value=f"{self.pred.means[feat]:.1f}")
            self.vars[feat] = var
            entry = ttk.Entry(left, textvariable=var, width=11, justify="right")
            entry.grid(row=3 * r + 1, column=0, sticky="w", pady=(2, 2))
            entry.bind("<Return>", lambda e, f=feat: self._on_entry(f))
            entry.bind("<FocusOut>", lambda e, f=feat: self._on_entry(f))
            ttk.Label(left, text=f"data range {lo:g} - {hi:g}", style="Muted.TLabel").grid(
                row=3 * r + 1, column=1, sticky="w", padx=(10, 0))
            scale = ttk.Scale(left, from_=lo, to=hi, orient="horizontal", length=260)
            scale.set(self.pred.means[feat])
            scale.configure(command=lambda v, f=feat: self._on_slide(f, v))  # attach after set()
            scale.grid(row=3 * r + 2, column=0, columnspan=2, sticky="we", pady=(0, 14))
            self.scales[feat] = scale

        ttk.Label(left, text="Model").grid(row=9, column=0, sticky="w", columnspan=2)
        self.model_var = tk.StringVar(value=self.pred.final)
        combo = ttk.Combobox(left, textvariable=self.model_var, values=self.pred.model_names,
                             state="readonly", width=34)
        combo.grid(row=10, column=0, columnspan=2, sticky="we", pady=(2, 14))
        combo.bind("<<ComboboxSelected>>", lambda e: self.predict())

        ttk.Button(left, text="Predict and add to history", style="Accent.TButton",
                   command=lambda: self.predict(record=True)).grid(row=11, column=0, columnspan=2, sticky="we")
        ttk.Button(left, text="Reset to average budgets", command=self.reset_inputs).grid(
            row=12, column=0, columnspan=2, sticky="we", pady=(8, 0))

        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        card = tk.Frame(right, bg=CARD, highlightbackground="#dde1e7", highlightthickness=1)
        card.grid(row=0, column=0, sticky="ew")
        tk.Label(card, text="Predicted " + self.pred.labels["Sales"], bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(14, 0))
        self.result_var = tk.StringVar(value="-")
        tk.Label(card, textvariable=self.result_var, bg=CARD, fg=ACCENT,
                 font=("Segoe UI", 40, "bold")).pack(anchor="w", padx=18)
        self.detail_var = tk.StringVar()
        tk.Label(card, textvariable=self.detail_var, bg=CARD, fg=INK, font=("Segoe UI", 10),
                 wraplength=560, justify="left").pack(anchor="w", padx=18)
        self.compare_var = tk.StringVar()
        tk.Label(card, textvariable=self.compare_var, bg=CARD, fg=MUTED, font=("Segoe UI", 9),
                 wraplength=560, justify="left").pack(anchor="w", padx=18, pady=(4, 0))
        self.gauge = tk.Canvas(card, width=560, height=78, bg=CARD, highlightthickness=0)
        self.gauge.pack(anchor="w", padx=18, pady=(6, 10))
        self.warn_var = tk.StringVar()
        ttk.Label(right, textvariable=self.warn_var, style="Warn.TLabel", wraplength=600,
                  justify="left").grid(row=1, column=0, sticky="w", pady=(8, 4))

        hist = ttk.LabelFrame(right, text="Prediction history", padding=8)
        hist.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        hist.columnconfigure(0, weight=1)
        hist.rowconfigure(0, weight=1)
        cols = list(self.pred.features) + ["Model", "Predicted sales"]
        self.history = ttk.Treeview(hist, columns=cols, show="headings", height=6)
        for c in cols:
            self.history.heading(c, text=c)
            self.history.column(c, width=250 if c == "Model" else 95, anchor="center")
        self.history.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(hist, orient="vertical", command=self.history.yview)
        self.history.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        bar = ttk.Frame(hist)
        bar.grid(row=1, column=0, columnspan=2, sticky="e", pady=(6, 0))
        ttk.Button(bar, text="Export to CSV...", command=self.export_history).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear", command=self.clear_history).pack(side="left")

    def _move_slider(self, feat, value):
        """Move a slider from code. ttk.Scale.set() fires the slider's command, which would overwrite
        what the user typed (rounding it or clamping it to the data range), so block that here."""
        self._syncing = True
        try:
            self.scales[feat].set(value)
        finally:
            self._syncing = False

    def _on_slide(self, feat, value):
        if self._syncing:
            return
        self.vars[feat].set(f"{float(value):.1f}")
        self.predict()

    def _on_entry(self, feat):
        try:
            v = float(self.vars[feat].get())
            lo, hi = self.pred.ranges[feat]
            self._move_slider(feat, min(max(v, lo), hi))  # slider only follows inside the data range
        except ValueError:
            pass
        self.predict()

    def reset_inputs(self):
        for feat in self.pred.features:
            self.vars[feat].set(f"{self.pred.means[feat]:.1f}")
            self._move_slider(feat, self.pred.means[feat])
        self.predict()

    def predict(self, record=False):
        try:
            values = self.pred.parse_inputs({f: v.get() for f, v in self.vars.items()})
        except InputError as exc:
            self.result_var.set("-")
            self.detail_var.set(str(exc))
            self.compare_var.set("")
            self.warn_var.set("")
            self.gauge.delete("all")
            return None

        name = self.model_var.get()
        y, clipped = self.pred.predict(name, values)
        rmse = self.pred.test_rmse(name)
        self.result_var.set(f"{y:,.2f}")
        self.detail_var.set(
            f"Typical error on unseen markets: about +/- {rmse:.2f}  (test RMSE of this model, "
            f"R² = {self.pred.test_r2(name):.3f}).")
        others = []
        for other in self.pred.model_names:
            if other != name:
                oy, _ = self.pred.predict(other, values)
                others.append(f"{other}: {oy:,.2f}")
        self.compare_var.set("For comparison  " + "   |   ".join(others))
        msgs = self.pred.range_warnings(values)
        if clipped:
            msgs.append("The model output was negative, so sales are shown as 0.")
        self.warn_var.set("\n".join(msgs))
        self._draw_gauge(y, rmse)
        if record:
            self.history.insert("", 0, values=[f"{values[f]:g}" for f in self.pred.features]
                                + [name, f"{y:,.2f}"])
        return y

    def _draw_gauge(self, y, rmse):
        c = self.gauge
        c.delete("all")
        lo, hi = self.pred.ranges["Sales"]
        x0, x1, ytrack = 20, 540, 40

        def px(v):
            return x0 + (x1 - x0) * (min(max(v, lo), hi) - lo) / (hi - lo)

        c.create_rectangle(x0, ytrack - 5, x1, ytrack + 5, fill="#e5e7eb", outline="")
        c.create_rectangle(px(y - rmse), ytrack - 5, px(y + rmse), ytrack + 5, fill="#a9c8f2", outline="")
        mean_x = px(self.pred.means["Sales"])
        c.create_line(mean_x, ytrack - 10, mean_x, ytrack + 10, fill=MUTED)
        c.create_text(mean_x, ytrack + 20, text=f"average {self.pred.means['Sales']:.1f}", fill=MUTED,
                      font=("Segoe UI", 8))
        mx = px(y)
        c.create_polygon(mx, ytrack - 7, mx - 7, ytrack - 20, mx + 7, ytrack - 20, fill=ACCENT, outline="")
        c.create_text(x0, ytrack + 20, text=f"{lo:g}", anchor="w", fill=MUTED, font=("Segoe UI", 8))
        c.create_text(x1, ytrack + 20, text=f"{hi:g}", anchor="e", fill=MUTED, font=("Segoe UI", 8))
        c.create_text(x0, 8, text="Position within the sales range of the dataset (light bar = typical error)",
                      anchor="w", fill=MUTED, font=("Segoe UI", 8))

    def export_history(self):
        rows = [self.history.item(i, "values") for i in self.history.get_children()]
        if not rows:
            messagebox.showinfo("History is empty", "Make a prediction with the button first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file", "*.csv")],
                                            initialfile="prediction_history.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(self.pred.features) + ["Model", "Predicted sales"])
            w.writerows(reversed(rows))  # oldest first
        messagebox.showinfo("Saved", f"History saved to:\n{path}")

    def clear_history(self):
        for i in self.history.get_children():
            self.history.delete(i)

    # ------------------------------------------------------------------ Dataset tab
    def _build_data_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        path = self.base / DATA_FILE
        try:
            header, rows = read_csv_table(path)
        except Exception as exc:
            ttk.Label(parent, text=f"Dataset file not found next to the app:\n{path}\n({exc})").grid(
                row=0, column=0, sticky="w")
            return
        ttk.Label(parent, text=f"{DATA_FILE}  -  {len(rows)} rows x {len(header)} columns").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=header, show="headings")
        for h in header:
            tree.heading(h, text=h)
            tree.column(h, width=170, anchor="center")
        for r in rows:
            tree.insert("", "end", values=r)
        tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")

        stats = ttk.LabelFrame(parent, text="Summary statistics", padding=8)
        stats.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        scols = ["Column", "Count", "Mean", "Min", "Max"]
        st = ttk.Treeview(stats, columns=scols, show="headings", height=len(header))
        for c in scols:
            st.heading(c, text=c)
            st.column(c, width=170 if c == "Column" else 100, anchor="center")
        for j, h in enumerate(header):
            col = [float(r[j]) for r in rows]
            st.insert("", "end", values=[h, len(col), f"{sum(col) / len(col):.2f}",
                                         f"{min(col):g}", f"{max(col):g}"])
        st.pack(fill="x")

    # ------------------------------------------------------------------ Model results tab
    def _build_model_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        try:
            with open(self.base / "results.json", encoding="utf-8") as f:
                res = json.load(f)
            header, rows = read_csv_table(self.base / "model_comparison.csv")
        except Exception as exc:
            ttk.Label(parent, text=f"results.json / model_comparison.csv not found ({exc}).\n"
                                   "Run python train_model.py first.").grid(row=0, column=0, sticky="w")
            return
        ttk.Label(parent, text="Model comparison (selection by 5-fold cross-validation on the training set; "
                               "metrics on the 20% test set)").grid(row=0, column=0, sticky="w", pady=(0, 6))
        tree = ttk.Treeview(parent, columns=header, show="headings", height=len(rows))
        for h in header:
            tree.heading(h, text=h)
            tree.column(h, width=250 if h == "Model" else 82, anchor="center")
        for r in rows:
            tree.insert("", "end", values=[r[0]] + [f"{float(v):.3f}" for v in r[1:]])
        tree.grid(row=1, column=0, sticky="ew")

        final = res["final_model"]
        t, b = res["test"][final], res["test"]["Multiple Linear Regression"]
        coef = res["baseline_coef"]
        cut = (1 - t["RMSE"] / b["RMSE"]) * 100
        lines = [
            f"Final model: {final}",
            f"Test set:  R² = {t['R2']:.3f}   RMSE = {t['RMSE']:.3f}   MAE = {t['MAE']:.3f}   "
            f"(plain multiple linear regression: R² = {b['R2']:.3f}, RMSE = {b['RMSE']:.3f})",
            "",
            "Baseline equation (original units):",
            f"Sales = {res['baseline_intercept']:.4f} + {coef['TV']:.5f} x TV + "
            f"{coef['Radio']:.5f} x Radio + {coef['Newspaper']:.5f} x Newspaper",
            "",
            "Reading the result: TV and Radio drive sales, Newspaper adds almost nothing. The straight-line "
            "model misses that TV and Radio reinforce each other, so adding squared and interaction terms "
            f"(still a linear regression, because it is linear in its coefficients) cuts the test RMSE by {cut:.0f}%.",
        ]
        box = tk.Frame(parent, bg=CARD, highlightbackground="#dde1e7", highlightthickness=1)
        box.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        tk.Label(box, text="\n".join(lines), bg=CARD, fg=INK, font=("Segoe UI", 10), justify="left",
                 wraplength=960, anchor="w").pack(anchor="w", padx=14, pady=12)

    # ------------------------------------------------------------------ Graphs tab
    def _build_plots_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        plot_dir = self.base / "plots"
        files = sorted(plot_dir.glob("*.png")) if plot_dir.exists() else []
        self.plot_files = files
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        ttk.Label(left, text="Select a graph").pack(anchor="w")
        self.plot_list = tk.Listbox(left, width=34, height=14, exportselection=False, font=("Segoe UI", 10),
                                    activestyle="none", selectbackground=ACCENT)
        self.plot_list.pack(fill="y", expand=True)
        for p in files:
            self.plot_list.insert("end", PLOT_TITLES.get(p.stem, p.stem))
        self.plot_list.bind("<<ListboxSelect>>", self._show_plot)
        self.plot_label = ttk.Label(parent, anchor="center",
                                    text="No graphs found - run train_model.py" if not files else "")
        self.plot_label.grid(row=0, column=1, sticky="nsew")
        if files:
            self.plot_list.selection_set(0)
            self._show_plot()

    def _show_plot(self, event=None):
        sel = self.plot_list.curselection()
        if not sel:
            return
        path = str(self.plot_files[sel[0]])
        if HAVE_PIL:
            img = Image.open(path)
            img.thumbnail((740, 560))
            self._photo = ImageTk.PhotoImage(img)
        else:  # plain Tk can only shrink by whole numbers
            photo = tk.PhotoImage(file=path)
            self._photo = photo.subsample(max(1, photo.width() // 740 + 1))
        self.plot_label.configure(image=self._photo)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
