"""Render the one-page default-case note; run from any working directory.

The text describes the defaults verified on 7 October 2026; update it when
changing the solver defaults. The companion .tex contains the same description.
"""

from pathlib import Path
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().with_name('plastic_plane_strain_case.pdf')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42})
W, H = 595.276, 841.89
fig = plt.figure(figsize=(W / 72, H / 72), dpi=144, facecolor='white')
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
left, width = 42, W - 84
y = 38
ink, accent, grey = '#172b3a', '#176475', '#55616b'


def label(text, x, top, size=9.4, color=ink, weight='normal', family=None):
    return fig.text(x / W, 1 - top / H, text, fontsize=size, color=color,
                    weight=weight, family=family, va='top', ha='left')


def para(text, size=9.4, gap=3, color=ink):
    global y
    font = FontProperties(family='DejaVu Sans', size=size)
    lines, current = [], ''
    for word in text.split():
        proposed = (current + ' ' + word).strip()
        pixels = renderer.get_text_width_height_descent(proposed, font, ismath=False)[0]
        if current and pixels / fig.dpi * 72 > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    lines.append(current)
    for line in lines:
        label(line, left, y, size=size, color=color)
        y += size * 1.35
    y += gap


def heading(text):
    global y
    y += 5
    label(text, left, y, size=10.5, color=accent, weight='bold')
    y += 17


def equation(text, height=25, size=11):
    global y
    label(text, left + 8, y, size=size)
    y += height


label('Plastic plane strain', left, y, size=22, weight='bold')
y += 29
label('Boundary conditions and solver settings', left, y, size=12, color=accent)
y += 21
label('Default run • coupling/plastic_plane_strain.py • 7 October 2026', left, y, size=8.5, color=grey)
y += 20

heading('Problem and prescribed displacement')
para('Small-strain, quasi-static J2 plasticity on a 1 m × 1 m square. Body force and all auxiliary sources are zero. Plane strain imposes zero total zz, xz and yz strains; full 3D stress and plastic history are retained, including nonzero out-of-plane stress.')
equation(r'$\nabla\!\cdot\!\sigma=0\ \mathrm{in}\ \Omega,\qquad u_x=0.004\,\ell\,x,\quad u_y=0\ \mathrm{on}\ \partial\Omega.$')
para('Both displacement components are prescribed at every boundary-face centre. The dimensionless load factor ℓ scales the absolute displacement:', gap=5)
rows = [('Boundary', 'Prescribed (uₓ, uᵧ)'), ('Left: x = 0', '(0, 0)'), ('Right: x = 1 m', '(0.004 ℓ m, 0)'), ('Bottom: y = 0; top: y = 1 m', '(0.004 ℓ x, 0)')]
row_h = 17
for i, (a, b) in enumerate(rows):
    fig.add_artist(Rectangle((left / W, 1 - (y + row_h) / H), width / W, row_h / H,
                            facecolor=('#e8f0f2' if i == 0 else '#f5f7f8'), edgecolor='white', linewidth=0.8))
    label(a, left + 7, y + 3, size=9, weight='bold' if i == 0 else 'normal')
    label(b, left + 261, y + 3, size=9, weight='bold' if i == 0 else 'normal')
    y += row_h
y += 6
para('This constrains transverse motion: top and bottom also move in x and carry reactions. No boundary tractions are prescribed. The homogeneous reference has εxx = 0.004 ℓ, εyy = εxy = 0; transverse stresses may be nonzero.')

heading('Material and spatial discretization')
para('E = 210 GPa, ν = 0.3, initial yield stress = 250 MPa. Linear mixed hardening uses H̄ = 1 GPa and θ = 0.4: the scalar laws are K(α) = 250 MPa + (0.4 GPa) α and H(α) = (0.6 GPa) α, where α is accumulated equivalent plastic strain.')
para('PorePy TPSA uses a fixed 3 × 3 Cartesian grid (9 cells, 36 unknowns): two displacement components, one rotation stress r and one total pressure p per cell. All [u, r, p] are solved together. Green–Gauss strain reconstruction feeds one 3D radial return map per cell. Stress corrections use fixed cell-to-face weights: 1/2 per neighbour inside; 1 at the boundary.')

heading('Loading and Newton solve')
para('Start from zero displacement, stress and history at ℓ = 0. Use 20 prescribed loading targets ℓ = 0.05, 0.10, …, 1, then unload to 0.99 and 0.98: 22 accepted steps. Peak εxx = 0.004 and right-edge displacement = 4 mm. Factors label load amplitude; no physical time integration is used.')
para('Each step starts from the previous converged [u, r, p]. Full Newton corrections use sparse LU (SciPy splu), with at most 20 corrections per step. The default Jacobian analytically differentiates the 3D return map and TPSA reconstruction/traction assembly. All three residual blocks must satisfy:')
equation(r'$J_k\,\Delta z_k=-R_k,\quad z_{k+1}=z_k+\Delta z_k,\qquad z=[u,r,p].$', height=23, size=10.5)
equation(r'$\|R_b^k\|_2\leq a_b+10^{-8}\|R_b^0\|_2,\quad (a_u,a_r,a_p)=(10^{-5},10^{-12},10^{-12}).$', height=25, size=10.2)
para('These are unweighted Euclidean norms of raw integrated residuals in each block’s SI units; reference norms are fixed at the step’s initial evaluation. There is no damping, line search or adaptive load stepping. Material history stays fixed during iteration and is accepted after Newton convergence; a failed solve preserves the preceding state.')
para('Local return-map Newton: at most 20 corrections; tolerance = 10⁻¹⁰ Pa + 10⁻¹² S, where S is the larger of the shifted trial deviatoric-stress norm and √(2/3) K(αn). Optional --jacobian finite-difference uses local forward differences with gradient perturbation 10⁻¹⁰ (--fd-step), with the same global assembly.', size=9.0)

heading('Checks and execution')
para('Every accepted step is checked for equilibrium, agreement with the independent homogeneous J2 reference, plane strain and elastic unloading. The default run first detects yield at step 8 (ℓ = 0.4). Newton convergence is printed; L2 field-error reporting is opt-in (--show-l2-errors).', size=9.0)
label('python -m coupling.plastic_plane_strain --no-export', left, y, size=8.8, family='DejaVu Sans Mono')
y += 17
para('Omit --no-export to write PNGs and VTK/PVD collections under results/.', size=8.6, color=grey, gap=0)
assert y < 800, f'Content extends to {y:.1f} pt'
label('Source: plastic_plane_strain.py; plane_strain.py; loading.py; newton.py;', left, 803, size=7.3, color=grey)
label('residual.py; jacobian.py; transfer.py; material_state.py; scalar_hardening.py; J2.py (repository files).', left, 814, size=7.3, color=grey)
metadata = {'Title': 'Plastic plane strain — boundary conditions and solver settings', 'Author': 'PorepyPlastic',
            'Subject': 'Default fully coupled TPSA J2 plasticity benchmark',
            'CreationDate': datetime(2026, 10, 7, tzinfo=timezone.utc)}
fig.savefig(OUT, format='pdf', metadata=metadata)
print(f'Wrote {OUT}; content ends at {y:.1f} / {H:.1f} pt')
plt.close(fig)
