import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])
os.makedirs("report/figs", exist_ok=True)
par = df[df.version == "par"].sort_values("threads")
n = par.threads.values

# Speedup
plt.figure()
plt.plot(n, par.speedup, "o-", label="msura")
plt.plot(n, n, "k--", label="ideal")
plt.xlabel("Threads")
plt.ylabel("Speedup vs sequencial")
plt.grid(True)
plt.legend()
plt.savefig("report/figs/speedup.pdf", bbox_inches="tight")

# Eficiencia
plt.figure()
plt.plot(n, par.efficiency, "o-")
plt.axhline(1, color="k", ls="--")
plt.xlabel("Threads")
plt.ylabel("Eficiència")
plt.ylim(0, 1.1)
plt.grid(True)
plt.savefig("report/figs/eficiencia.pdf", bbox_inches="tight")

# Tiempo
plt.figure()
plt.plot(n, par.time, "o-")
plt.xlabel("Threads")
plt.ylabel("Temps execució (s)")
plt.grid(True)
plt.savefig("report/figs/temps.pdf", bbox_inches="tight")
