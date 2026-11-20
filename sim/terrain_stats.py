"""Arazi istatistikleri: golgelik/bataklik oranlari ve insanin yeri."""
import numpy as np
import env as E

e = E.SearchEnv(300, cells=100)
cs = e.canopy.reshape(300, -1)
print(f"golgelik>0.5 orani: %{100*(cs > 0.5).mean():.1f}  ortalama golgelik: {cs.mean():.2f}")
print(f"bataklik orani:     %{100*e.swamp.mean():.1f}")
ij = np.stack([(e.py / E.CELL).astype(int), (e.px / E.CELL).astype(int)], 1)
pc = e.canopy[np.arange(300), ij[:, 0], ij[:, 1]]
print(f"insanin hucresinde ort. golgelik: {pc.mean():.2f}")
print(f"insan bataklikta: %{100*e.swamp[np.arange(300), ij[:,0], ij[:,1]].mean():.1f}")
