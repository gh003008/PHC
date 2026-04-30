"""Bidirectional Tkinter panel for phc_walk_demo.py.

Reads /tmp/phc_walk_state.json (status display).
Writes /tmp/phc_walk_input.json on Apply / button / slider release.

Layout:
  - status block: v_cmd_target, v_cmd_ramped, v_actual, active_clip, v_natural,
                  retime_ratio, step, paused
  - Entry widget for direct numeric v_cmd input + Apply button
  - Slider (Tk.Scale) over [V_MIN, V_MAX]
  - Reset / Pause buttons

Launched automatically by phc_walk_demo.py as a subprocess. Standalone:
  python scripts/phc_walk_demo_panel.py
"""
import json
import os
import tkinter as tk
from tkinter import ttk

STATE_PATH = "/tmp/phc_walk_state.json"
INPUT_PATH = "/tmp/phc_walk_input.json"

V_MIN = 0.76   # mirrors phc_walk_demo.py constants
V_MAX = 1.23


def lerp_color(t):
    t = max(0.0, min(1.0, t))
    r = int(255 * t)
    g = 0
    b = int(255 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_input(payload: dict):
    """Atomic write of the input file so the demo doesn't see a partial JSON."""
    tmp = INPUT_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, INPUT_PATH)
    except Exception as e:
        print(f"[panel] write failed: {e}")


class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PHC Walk Demo — v_cmd")
        sw = self.root.winfo_screenwidth()
        w, h = 360, 480
        x = sw - w - 30
        y = 30
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg="#111")
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        big = ("DejaVu Sans Mono", 26, "bold")
        med = ("DejaVu Sans", 14)
        small = ("DejaVu Sans", 11)

        self.title_lbl = tk.Label(self.root, text="PHC Walk Demo",
                                  font=("DejaVu Sans", 16, "bold"),
                                  fg="#aaa", bg="#111")
        self.title_lbl.pack(pady=(8, 4))

        # Big v_cmd_ramped display
        self.value_lbl = tk.Label(self.root, text="--.--", font=big,
                                  fg="#888", bg="#111")
        self.value_lbl.pack()
        self.unit_lbl = tk.Label(self.root, text="m/s", font=small,
                                 fg="#888", bg="#111")
        self.unit_lbl.pack()

        # Status block
        self.target_lbl = tk.Label(self.root, text="target —", font=med,
                                   fg="#ddd", bg="#111")
        self.target_lbl.pack(pady=(6, 0))
        self.actual_lbl = tk.Label(self.root, text="actual —", font=med,
                                   fg="#ddd", bg="#111")
        self.actual_lbl.pack()
        self.clip_lbl = tk.Label(self.root, text="clip —", font=small,
                                 fg="#aaa", bg="#111")
        self.clip_lbl.pack(pady=(4, 0))
        self.ratio_lbl = tk.Label(self.root, text="ratio —", font=small,
                                  fg="#aaa", bg="#111")
        self.ratio_lbl.pack()
        self.step_lbl = tk.Label(self.root, text="step —", font=small,
                                 fg="#666", bg="#111")
        self.step_lbl.pack()
        self.pause_lbl = tk.Label(self.root, text="", font=("DejaVu Sans", 14, "bold"),
                                  fg="#ffa500", bg="#111")
        self.pause_lbl.pack(pady=(2, 6))

        # Numeric input
        entry_frame = tk.Frame(self.root, bg="#111")
        entry_frame.pack(pady=(4, 0))
        tk.Label(entry_frame, text="Set v_cmd:", font=small, fg="#aaa", bg="#111"
                 ).grid(row=0, column=0, padx=4)
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(entry_frame, textvariable=self.entry_var, width=8,
                              font=med, bg="#222", fg="#fff", insertbackground="#fff")
        self.entry.grid(row=0, column=1, padx=4)
        self.entry.bind("<Return>", lambda e: self.on_apply())
        tk.Button(entry_frame, text="Apply", font=small, command=self.on_apply,
                  bg="#333", fg="#fff").grid(row=0, column=2, padx=4)

        # Slider
        self.slider_var = tk.DoubleVar(value=1.0)
        self.slider = tk.Scale(self.root, variable=self.slider_var,
                               from_=V_MIN, to=V_MAX, resolution=0.01,
                               orient=tk.HORIZONTAL, length=300,
                               bg="#111", fg="#ddd", troughcolor="#333",
                               highlightthickness=0,
                               command=self.on_slider)
        self.slider.pack(pady=(8, 0))

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#111")
        btn_frame.pack(pady=(8, 0))
        tk.Button(btn_frame, text="Reset", font=small, width=8,
                  command=self.on_reset, bg="#333", fg="#fff"
                  ).grid(row=0, column=0, padx=4)
        tk.Button(btn_frame, text="Pause", font=small, width=8,
                  command=self.on_pause, bg="#333", fg="#fff"
                  ).grid(row=0, column=1, padx=4)
        tk.Button(btn_frame, text="Quit", font=small, width=8,
                  command=self.on_quit, bg="#600", fg="#fff"
                  ).grid(row=0, column=2, padx=4)

        self._slider_dragging = False
        self.tick()

    def on_apply(self):
        try:
            v = float(self.entry_var.get())
        except ValueError:
            return
        v = max(V_MIN, min(V_MAX, v))
        self.slider_var.set(v)
        write_input({"v_cmd_target": v})

    def on_slider(self, _val):
        v = float(self.slider_var.get())
        write_input({"v_cmd_target": v})

    def on_reset(self):
        write_input({"reset": True})

    def on_pause(self):
        write_input({"pause_toggle": True})

    def on_quit(self):
        write_input({"quit": True})
        self.root.after(200, self.root.destroy)

    def tick(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as f:
                    s = json.load(f)
                v_ramped = float(s.get("v_cmd_ramped", 0.0))
                v_target = float(s.get("v_cmd_target", 0.0))
                v_actual = float(s.get("v_actual", 0.0))
                clip = int(s.get("active_clip", 0))
                v_natural = float(s.get("v_natural", 0.0))
                ratio = float(s.get("retime_ratio", 1.0))
                step = int(s.get("step", 0))
                paused = bool(s.get("paused", False))
                vmin = float(s.get("v_min", V_MIN))
                vmax = float(s.get("v_max", V_MAX))

                t = (v_ramped - vmin) / max(vmax - vmin, 1e-6)
                color = lerp_color(t)
                self.value_lbl.config(text=f"{v_ramped:.3f}", fg=color)
                self.unit_lbl.config(fg=color)
                self.target_lbl.config(text=f"target  {v_target:.3f} m/s")
                self.actual_lbl.config(text=f"actual  {v_actual:.3f} m/s")
                clip_letter = "ABC"[clip] if 0 <= clip < 3 else "?"
                self.clip_lbl.config(text=f"clip {clip_letter} (natural {v_natural:.3f})")
                self.ratio_lbl.config(text=f"retime ratio  {ratio:.3f}x")
                self.step_lbl.config(text=f"step  {step}")
                self.pause_lbl.config(text="PAUSED" if paused else "")
        except Exception:
            pass
        self.root.after(100, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Panel().run()
