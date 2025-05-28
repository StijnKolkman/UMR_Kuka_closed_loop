import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import numpy as np

root = tk.Tk()
root.geometry("800x600")

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

fig = Figure(figsize=(6, 5))
ax3d = fig.add_subplot(111, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Plot a sample 3D trajectory
x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)
ax3d.plot(x, y, z)

ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_title("Sample 3D plot")

canvas.draw()

root.mainloop()
