"""
UML-діаграма класів проєкту PollutionExperiment.
Пастельний стиль (А4 portrait, 210×297 мм).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ==============================================================
# Опис класів
# ==============================================================
CLASSES = {
    'Pesticide': {
        'h': '#6FA9C0', 'b': '#D6EAF2',
        'attrs': [
            '-  name: str',
            '-  k_oc: float',
            '-  half_life: float',
            '-  gdk: float',
            '-  decay_rate: float',
        ],
        'methods': [
            '+  __init__(name, k_oc, …)',
        ],
        'note': '',
    },
    'SoilProfile': {
        'h': '#6FBC8B', 'b': '#D5ECD9',
        'attrs': [
            '-  foc: float',
            '-  bulk_density: float',
            '-  porosity: float',
            '-  mixing_depth_m: float',
        ],
        'methods': [
            '+  compute_retardation(k_oc)',
            '+  compute_mobile_fraction(k_oc)',
        ],
        'note': '',
    },
    'AgriculturalField': {
        'h': '#B5C97D', 'b': '#E5EDC8',
        'attrs': [
            '-  area_ha: float',
            '-  soil: SoilProfile',
            '-  cn: float',
            '-  s_max: float',
        ],
        'methods': [
            '+  calculate_runoff_mm(precip)',
        ],
        'note': '',
    },
    'River': {
        'h': '#D88787', 'b': '#F2D2D2',
        'attrs': [
            '-  width: float',
            '-  depth: float',
            '-  velocity: float',
            '-  base_flow: float',
            '-  dispersion: float',
        ],
        'methods': [],
        'note': '',
    },
    'BufferZone': {
        'h': '#7DB7A5', 'b': '#D7ECE5',
        'attrs': [
            '-  width_m: float',
            '-  slope_pct: float',
            '-  vegetation_type: str',
            '-  infiltration_rate_mm_h: float',
            '-  _alpha: float',
        ],
        'methods': [
            '+  compute_delivery(mass, decay)',
        ],
        'note': 'Retention & delayed transport',
    },
    'SimulationEngine': {
        'h': '#9C7BB5', 'b': '#E0D2EA',
        'attrs': [
            '-  field: AgriculturalField',
            '-  pesticide: Pesticide',
            '-  river: River',
            '-  buffer: BufferZone',
        ],
        'methods': [
            '+  compute_field_residue(dose, day)',
            '+  compute_daily_export(residue, precip)',
            '+  compute_buffer_delivery(export)',
            '+  solve_river_transport(mass, len, dur)',
            '+  run_temporal_simulation(dose, precip[])',
            '+  sensitivity_analysis(param, values)',
        ],
        'note': '',
    },
    'Visualizer': {
        'h': '#C8A04F', 'b': '#F2E0B0',
        'attrs': [
            '[static methods only]',
        ],
        'methods': [
            '+  plot_temporal_simulation()',
            '+  plot_river_plume()',
            '+  plot_sensitivity()',
        ],
        'note': '',
    },
    'UncertainParameter': {
        'h': '#5BAA72', 'b': '#C2DEC9',
        'attrs': [
            '-  name: str',
            '-  mean: float',
            '-  std: float',
            '-  dist: str',
            '-  low / high: float',
        ],
        'methods': [
            '+  sample(n)',
        ],
        'note': '',
    },
    'MonteCarloEngine': {
        'h': '#C7796A', 'b': '#EDC4BB',
        'attrs': [
            '-  n_iterations: int',
            '-  results: dict',
            '-  params_history: dict',
        ],
        'methods': [
            '+  run(uncertain, fixed)',
            '+  compute_sensitivity_indices()',
        ],
        'note': '',
    },
    'MonteCarloVisualizer': {
        'h': '#7BA199', 'b': '#C5DAD4',
        'attrs': [
            '[static methods only]',
        ],
        'methods': [
            '+  plot_full_report()',
            '+  plot_convergence()',
        ],
        'note': '',
    },
}


# ==============================================================
# Сторінка
# ==============================================================
W, H = 8.27, 11.69
MARGIN = 0.30

FS_TITLE = 16
FS_HEADER = 12
FS_BODY = 9.2
FS_NOTE = 7.8
HEADER_H = 0.50
LINE_H = 0.22


def text_block_h(cls):
    n_attr = sum(s.count('\n') + 1 for s in cls['attrs']) if cls['attrs'] else 0
    n_meth = sum(s.count('\n') + 1 for s in cls['methods']) if cls['methods'] else 0
    h_attr = LINE_H * max(1, n_attr) + 0.12
    h_meth = (LINE_H * n_meth + 0.12) if n_meth else 0
    h_note = 0.24 if cls.get('note') else 0
    return HEADER_H + h_attr + h_meth + h_note


# ==============================================================
# Розміщення
# ==============================================================
LAYOUT = {}

# --- Рядок 1: 4 entity-класи (вузькі) ---
ENT_W = 1.45
ent_gap = (W - 2 * MARGIN - 5 * ENT_W) / 4
top_row_top = 11.55
ent_h = max(text_block_h(CLASSES[n]) for n in
            ['Pesticide', 'SoilProfile', 'AgriculturalField', 'River', 'BufferZone'])
for i, name in enumerate([
    'Pesticide',
    'SoilProfile',
    'AgriculturalField',
    'BufferZone',
    'River'
]):
    x = MARGIN + i * (ENT_W + ent_gap)
    LAYOUT[name] = (x, top_row_top - ent_h, ENT_W, ent_h)

# --- Рядок 2: SimulationEngine (ліво) + Visualizer (право) ---
se_h = text_block_h(CLASSES['SimulationEngine'])
viz_h = text_block_h(CLASSES['Visualizer'])
SE_W = 4.30
VIZ_W = W - 2 * MARGIN - SE_W - 0.30
row2_top = 8.40
LAYOUT['SimulationEngine'] = (MARGIN, row2_top - se_h, SE_W, se_h)
LAYOUT['Visualizer'] = (MARGIN + SE_W + 0.30,
                        row2_top - viz_h - 0.10,
                        VIZ_W, viz_h)

# --- Рядок 3: MonteCarloEngine (ліво) + UncertainParameter (право) ---
mc_h = text_block_h(CLASSES['MonteCarloEngine'])
up_h = text_block_h(CLASSES['UncertainParameter'])
MC_W = 3.55
UP_W = W - 2 * MARGIN - MC_W - 0.30
row3_top = 4.20
LAYOUT['MonteCarloEngine'] = (MARGIN, row3_top - mc_h, MC_W, mc_h)
LAYOUT['UncertainParameter'] = (MARGIN + MC_W + 0.30,
                                row3_top - up_h,
                                UP_W, up_h)

# --- Рядок 4: MonteCarloVisualizer (центр-ліво) ---
mv_h = text_block_h(CLASSES['MonteCarloVisualizer'])
MV_W = 3.55
row4_top = 1.50
LAYOUT['MonteCarloVisualizer'] = (MARGIN, row4_top - mv_h, MV_W, mv_h)


# ==============================================================
# Малювання
# ==============================================================
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')

def draw_box(name):
    cls = CLASSES[name]
    x, y, w, h = LAYOUT[name]

    # Тінь
    ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.05), w, h,
                                boxstyle="round,pad=0.005,rounding_size=0.06",
                                linewidth=0, facecolor='#00000018'))

    # Тіло
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.005,rounding_size=0.06",
                                linewidth=0.8, edgecolor='#444', facecolor=cls['b']))

    # Заголовок
    ax.add_patch(FancyBboxPatch((x, y + h - HEADER_H), w, HEADER_H,
                                boxstyle="round,pad=0.005,rounding_size=0.06",
                                linewidth=0, facecolor=cls['h']))
    ax.text(x + w / 2, y + h - HEADER_H / 2, name,
            ha='center', va='center', fontsize=FS_HEADER,
            color='white', fontweight='bold')

    n_attr = sum(s.count('\n') + 1 for s in cls['attrs']) if cls['attrs'] else 0
    h_attr = LINE_H * max(1, n_attr) + 0.12
    sep1_y = y + h - HEADER_H
    sep2_y = sep1_y - h_attr

    if cls['methods']:
        ax.plot([x + 0.10, x + w - 0.10], [sep2_y, sep2_y],
                color='#888', linewidth=0.5)

    # Атрибути
    cur = sep1_y - 0.12
    for line in cls['attrs']:
        for sub in line.split('\n'):
            stripped = sub.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                ax.text(x + w / 2, cur, stripped,
                        ha='center', va='top', fontsize=FS_NOTE,
                        family='DejaVu Sans', style='italic', color='#555')
            else:
                ax.text(x + 0.18, cur, sub, ha='left', va='top',
                        fontsize=FS_BODY, family='DejaVu Sans')
            cur -= LINE_H

    # Методи
    cur = sep2_y - 0.12
    for line in cls['methods']:
        for sub in line.split('\n'):
            stripped = sub.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                ax.text(x + 0.36, cur, stripped,
                        ha='left', va='top', fontsize=FS_NOTE,
                        family='DejaVu Sans', style='italic', color='#555')
            else:
                ax.text(x + 0.18, cur, sub, ha='left', va='top',
                        fontsize=FS_BODY, family='DejaVu Sans')
            cur -= LINE_H

    # Нижня позначка
    if cls.get('note'):
        ax.text(x + w / 2, y + 0.10, cls['note'],
                ha='center', va='bottom', fontsize=FS_NOTE,
                style='italic', color='#555')


for name in CLASSES:
    draw_box(name)


# ==============================================================
# Зв'язки
# ==============================================================
def anchor(name, side, frac=0.5):
    x, y, w, h = LAYOUT[name]
    if side == 'top':    return (x + w * frac, y + h)
    if side == 'bottom': return (x + w * frac, y)
    if side == 'left':   return (x, y + h * frac)
    if side == 'right':  return (x + w, y + h * frac)


def draw_arrow(src, dst, kind='solid', color='#3D4F73', label='',
               label_pos=0.5, label_offset=(0, 0), curve=0.0):
    if kind == 'dashed':
        style = dict(linewidth=1.5, color='#B0413E',
                     linestyle=(0, (5, 3)), arrowstyle='-|>',
                     mutation_scale=15, shrinkA=2, shrinkB=2)
    else:
        style = dict(linewidth=1.5, color=color,
                     arrowstyle='-|>', mutation_scale=15,
                     shrinkA=2, shrinkB=2)
    arrow = FancyArrowPatch(src, dst,
                            connectionstyle=f"arc3,rad={curve}", **style)
    ax.add_patch(arrow)
    if label:
        mx = src[0] + (dst[0] - src[0]) * label_pos + label_offset[0]
        my = src[1] + (dst[1] - src[1]) * label_pos + label_offset[1]
        col = '#B0413E' if kind == 'dashed' else color
        ax.text(mx, my, label, ha='center', va='center',
                fontsize=8.5, style='italic',
                color=col, fontweight='bold' if kind == 'dashed' else 'normal',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor='none', alpha=0.95))


# AgriculturalField has-a SoilProfile
draw_arrow(anchor('AgriculturalField', 'left'),
           anchor('SoilProfile', 'right'),
           kind='solid', color='#3D7B57', label='has-a',
           label_offset=(0, -0.14))

# BufferZone → River
# Буферна зона розташована між полем та річкою
draw_arrow(anchor('BufferZone', 'right'),
           anchor('River', 'left'),
           kind='dashed',
           label='filters runoff',
           label_offset=(0, 0.16),
           curve=0.08)

# SimulationEngine → 3 прямі атрибути (Pesticide, AgriculturalField, River)
# SoilProfile доступний транзитивно через AgriculturalField — зв'язок не дублюємо
draw_arrow(anchor('SimulationEngine', 'top', frac=0.08),
           anchor('Pesticide', 'bottom'))

draw_arrow(anchor('SimulationEngine', 'top', frac=0.33),
           anchor('AgriculturalField', 'bottom'))

draw_arrow(anchor('SimulationEngine', 'top', frac=0.58),
           anchor('BufferZone', 'bottom'))

draw_arrow(anchor('SimulationEngine', 'top', frac=0.86),
           anchor('River', 'bottom'))

# Visualizer → SimulationEngine
draw_arrow(anchor('Visualizer', 'left'),
           anchor('SimulationEngine', 'right', frac=0.30),
           kind='dashed', label='uses', label_offset=(0, 0.20))

# MonteCarloEngine → UncertainParameter (MC приймає UP як вхід)
draw_arrow(anchor('MonteCarloEngine', 'right'),
           anchor('UncertainParameter', 'left'),
           kind='dashed', label='uses', label_offset=(0, 0.18))

# MonteCarloEngine → SimulationEngine
draw_arrow(anchor('MonteCarloEngine', 'top', frac=0.30),
           anchor('SimulationEngine', 'bottom', frac=0.18),
           kind='dashed', label='uses',
           label_offset=(-0.30, 0))

# MonteCarloVisualizer → MonteCarloEngine
draw_arrow(anchor('MonteCarloVisualizer', 'top'),
           anchor('MonteCarloEngine', 'bottom'),
           kind='dashed', label='uses', label_offset=(0.20, 0))


# ==============================================================
os.makedirs('output', exist_ok=True)
out_path = 'output/class_diagram.png'
fig.savefig(out_path, dpi=200, bbox_inches=None, pad_inches=0)
print(f'Збережено: {out_path}')
plt.close()
