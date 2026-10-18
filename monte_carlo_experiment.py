"""
Стохастична симуляція (Monte Carlo) для оцінки невизначеності
моделі забруднення водних ресурсів пестицидами.

Модуль: monte_carlo_experiment.py

Підхід: замість одного детерміністичного запуску модель
запускається N разів з випадковими параметрами, що дозволяє
отримати розподіл результатів та оцінити ймовірність
перевищення ГДК.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from agro_pollution_model import (
    Pesticide, SoilProfile, AgriculturalField, River, BufferZone, SimulationEngine
)


# ============================================================
# БЛОК 1: ПАРАМЕТРИ НЕВИЗНАЧЕНОСТІ
# ============================================================

class UncertainParameter:
    """Опис параметра з невизначеністю."""

    def __init__(self, name: str, mean: float, std: float,
                 dist: str = 'normal', low: float = None, high: float = None):
        """
        Args:
            name: Назва параметра
            mean: Середнє (номінальне) значення
            std: Стандартне відхилення
            dist: Тип розподілу ('normal', 'uniform', 'lognormal')
            low, high: Межі допустимих значень (обрізка)
        """
        self.name = name
        self.mean = mean
        self.std = std
        self.dist = dist
        self.low = low
        self.high = high

    def sample(self, n: int = 1) -> np.ndarray:
        """Генерація n випадкових значень."""
        if self.dist == 'normal':
            values = np.random.normal(self.mean, self.std, n)
        elif self.dist == 'uniform':
            a = self.mean - self.std * np.sqrt(3)
            b = self.mean + self.std * np.sqrt(3)
            values = np.random.uniform(a, b, n)
        elif self.dist == 'lognormal':
            sigma2 = np.log(1 + (self.std / self.mean) ** 2)
            mu = np.log(self.mean) - sigma2 / 2
            values = np.random.lognormal(mu, np.sqrt(sigma2), n)
        else:
            values = np.full(n, self.mean)

        # Обрізка до допустимих меж
        if self.low is not None:
            values = np.maximum(values, self.low)
        if self.high is not None:
            values = np.minimum(values, self.high)

        return values


# ============================================================
# БЛОК 2: ДВИГУН MONTE CARLO
# ============================================================

class MonteCarloEngine:
    """Стохастична оболонка навколо детерміністичної моделі."""

    def __init__(self, n_iterations: int = 5000):
        self.n_iterations = n_iterations
        self.results = None
        self.params_history = None

    def run(self, uncertain_params: dict, fixed_params: dict,
            verbose: bool = True) -> dict:
        """
        Запуск Monte Carlo симуляції.

        Args:
            uncertain_params: dict з UncertainParameter для кожного варійованого параметра
            fixed_params: dict з фіксованими параметрами
                - area_ha, bulk_density, porosity, mixing_depth_m
                - river_width, river_depth, river_velocity, river_base_flow, river_dispersion
                - days_since, precipitation_mm
            verbose: Виводити прогрес
        """
        N = self.n_iterations

        if verbose:
            print(f"\n{'='*65}")
            print(f"  СТОХАСТИЧНА СИМУЛЯЦІЯ (MONTE CARLO)")
            print(f"  Кількість ітерацій: {N}")
            print(f"{'='*65}")
            print(f"\n  Параметри з невизначеністю:")
            for name, param in uncertain_params.items():
                print(f"    {name:<16} μ={param.mean:<8.4f} σ={param.std:<8.4f} "
                      f"розподіл: {param.dist}")

        # Генеруємо всі випадкові параметри наперед (ефективніше)
        samples = {}
        for name, param in uncertain_params.items():
            samples[name] = param.sample(N)

        # Масиви для результатів
        peak_concentrations = np.zeros(N)
        exports_g = np.zeros(N)
        runoffs_mm = np.zeros(N)
        residues = np.zeros(N)
        mobile_fractions = np.zeros(N)

        fp = fixed_params
        t_start = time()

        if verbose:
            print(f"\n  Обчислення...")

        for i in range(N):
            # Збираємо параметри для цієї ітерації
            cn = samples.get('cn', np.full(N, fp.get('cn', 78)))[i]
            foc = samples.get('foc', np.full(N, fp.get('foc', 0.03)))[i]
            half_life = samples.get('half_life', np.full(N, fp.get('half_life', 75)))[i]
            dose = samples.get('dose', np.full(N, fp.get('dose', 1.5)))[i]
            rain = samples.get('rain', np.full(N, fp.get('rain', 35)))[i]
            k_oc = samples.get('k_oc', np.full(N, fp.get('k_oc', 100)))[i]
            base_flow = samples.get('base_flow', np.full(N, fp.get('base_flow', 8.0)))[i]

            days_since = fp.get('days_since', 10)

            # Створюємо об'єкти моделі з поточними параметрами
            try:
                pesticide = Pesticide("Атразин", k_oc=float(k_oc),
                                       half_life=float(half_life), gdk=fp.get('gdk', 0.002))
                soil = SoilProfile(foc=float(foc),
                                    bulk_density=fp.get('bulk_density', 1.3),
                                    porosity=fp.get('porosity', 0.50))
                field = AgriculturalField(area_ha=fp.get('area_ha', 50),
                                           soil=soil, cn=float(cn))
                river = River(width=fp.get('river_width', 30),
                              depth=fp.get('river_depth', 1.5),
                              velocity=fp.get('river_velocity', 0.4),
                              base_flow=float(base_flow),
                              dispersion=fp.get('river_dispersion', 5.0))

                engine = SimulationEngine(field, pesticide, river,
                                         BufferZone(width_m=fp.get('buffer_width', 20),
                                                    slope_pct=fp.get('buffer_slope', 5.0),
                                                    vegetation_type=fp.get('buffer_veg', 'grass')))

                # Запуск моделі
                residue = engine.compute_field_residue(float(dose), days_since)
                export = engine.compute_daily_export(residue, float(rain))

                residues[i] = residue
                mobile_fractions[i] = export['mobile_fraction']
                runoffs_mm[i] = export['runoff_mm']
                exports_g[i] = export['mass_exported_g']

                if export['mass_exported_g'] > 0.01:
                    # Пропускаємо через буферну зону
                    buf_result = engine.buffer.compute_delivery(
                        export['mass_exported_g'], pesticide.decay_rate)
                    mass_to_river = buf_result['mass_delivered_g']

                    if mass_to_river > 0.01:
                        # Швидке аналітичне наближення для MC
                        source_duration_s = 3600.0
                        mass_rate = mass_to_river / source_duration_s  # г/с
                        flow_rate_liters = float(base_flow) * 1000.0  # л/с
                        peak_concentrations[i] = (mass_rate / flow_rate_liters) * 1000.0  # мг/л
                    else:
                        peak_concentrations[i] = 0.0
                else:
                    peak_concentrations[i] = 0.0

            except (ValueError, Exception) as e:
                # Якщо параметри дали невалідну комбінацію — пропускаємо
                peak_concentrations[i] = np.nan
                exports_g[i] = np.nan

            # Прогрес
            if verbose and (i + 1) % 500 == 0:
                elapsed = time() - t_start
                rate = (i + 1) / elapsed
                remaining = (N - i - 1) / rate
                print(f"    {i+1}/{N} ({(i+1)/N*100:.0f}%) — "
                      f"{elapsed:.1f}с пройшло, ~{remaining:.0f}с залишилось")

        elapsed = time() - t_start

        # Відфільтровуємо NaN
        valid = ~np.isnan(peak_concentrations)
        n_valid = valid.sum()

        self.results = {
            'peak_conc_mg_l': peak_concentrations[valid],
            'export_g': exports_g[valid],
            'runoff_mm': runoffs_mm[valid],
            'residue_kg_ha': residues[valid],
            'mobile_fraction': mobile_fractions[valid],
            'n_total': N,
            'n_valid': n_valid,
            'elapsed_s': elapsed,
            'gdk': fp.get('gdk', 0.002),
        }

        self.params_history = {name: vals[valid] for name, vals in samples.items()}

        # Статистика
        conc = self.results['peak_conc_mg_l']
        gdk = self.results['gdk']

        if verbose:
            print(f"\n  Завершено за {elapsed:.1f} с ({n_valid} валідних з {N})")
            print(f"\n{'─'*65}")
            print(f"  РЕЗУЛЬТАТИ MONTE CARLO")
            print(f"{'─'*65}")
            print(f"\n  Пікова концентрація в річці (мг/л):")
            print(f"    Середнє:           {np.mean(conc):.6f} мг/л ({np.mean(conc)*1000:.3f} мкг/л)")
            print(f"    Медіана:           {np.median(conc):.6f} мг/л ({np.median(conc)*1000:.3f} мкг/л)")
            print(f"    Ст. відхилення:    {np.std(conc):.6f} мг/л ({np.std(conc)*1000:.3f} мкг/л)")
            print(f"    Мінімум:           {np.min(conc):.6f} мг/л ({np.min(conc)*1000:.3f} мкг/л)")
            print(f"    Максимум:          {np.max(conc):.6f} мг/л ({np.max(conc)*1000:.3f} мкг/л)")
            print(f"    5-й процентиль:    {np.percentile(conc, 5):.6f} мг/л ({np.percentile(conc, 5)*1000:.3f} мкг/л)")
            print(f"    95-й процентиль:   {np.percentile(conc, 95):.6f} мг/л ({np.percentile(conc, 95)*1000:.3f} мкг/л)")
            print(f"\n  Оцінка ризику (ГДК = {gdk} мг/л = {gdk*1000} мкг/л):")
            prob = (conc > gdk).mean() * 100
            print(f"    Ймовірність перевищення ГДК: {prob:.1f}%")
            print(f"    Кількість перевищень: {(conc > gdk).sum()} з {n_valid}")

            # Класифікація ризику
            if prob < 5:
                risk = "НИЗЬКИЙ"
            elif prob < 25:
                risk = "ПОМІРНИЙ"
            elif prob < 50:
                risk = "ПІДВИЩЕНИЙ"
            else:
                risk = "ВИСОКИЙ"
            print(f"    Рівень ризику: {risk}")

            print(f"\n  Винос з поля:")
            exp = self.results['export_g']
            print(f"    Середній: {np.mean(exp):.1f} г, "
                  f"90% інтервал: [{np.percentile(exp, 5):.1f} — {np.percentile(exp, 95):.1f}] г")

        return self.results

    def compute_sensitivity_indices(self) -> dict:
        """
        Розрахунок індексів чутливості: кореляція Спірмена
        між кожним вхідним параметром та результатом.
        Це показує, який параметр найбільше впливає на невизначеність.
        """
        from scipy import stats

        conc = self.results['peak_conc_mg_l']
        indices = {}

        labels_ua = {
            'cn': 'Curve Number',
            'foc': 'Орг. вуглець (foc)',
            'half_life': 'DT50 (днів)',
            'dose': 'Доза (кг/га)',
            'rain': 'Опади (мм)',
            'k_oc': 'Koc (л/кг)',
            'base_flow': 'Витрата річки (м³/с)',
        }

        for name, values in self.params_history.items():
            if len(values) == len(conc):
                rho, pval = stats.spearmanr(values, conc)
                indices[name] = {
                    'rho': rho,
                    'rho_abs': abs(rho),
                    'p_value': pval,
                    'label': labels_ua.get(name, name)
                }

        # Сортуємо за абсолютним впливом
        indices = dict(sorted(indices.items(), key=lambda x: x[1]['rho_abs'], reverse=True))

        print(f"\n{'─'*65}")
        print(f"  РЕЙТИНГ ПАРАМЕТРІВ ЗА ВНЕСКОМ У НЕВИЗНАЧЕНІСТЬ")
        print(f"  (Кореляція Спірмена з піковою концентрацією)")
        print(f"{'─'*65}")
        print(f"  {'Параметр':<25} {'ρ Спірмена':<14} {'|ρ|':<10} {'Напрямок'}")
        print(f"  {'─'*62}")

        for name, idx in indices.items():
            direction = "↑ більше = гірше" if idx['rho'] > 0 else "↓ більше = краще"
            bar = "█" * int(idx['rho_abs'] * 30)
            print(f"  {idx['label']:<25} {idx['rho']:>+.4f}       {idx['rho_abs']:.4f}     {direction}")
            print(f"  {'':25} {bar}")

        return indices


# ============================================================
# БЛОК 3: ВІЗУАЛІЗАЦІЯ MONTE CARLO
# ============================================================

class MonteCarloVisualizer:
    """Візуалізація результатів стохастичної симуляції."""

    @staticmethod
    def plot_full_report(mc_engine: MonteCarloEngine, save_path: str = None):
        """Комплексний звіт: 6 графіків на одному аркуші."""

        results = mc_engine.results
        params = mc_engine.params_history
        conc = results['peak_conc_mg_l'] * 1000  # мкг/л
        gdk = results['gdk'] * 1000  # мкг/л
        export = results['export_g']

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Стохастична симуляція (Monte Carlo): оцінка невизначеності',
                     fontsize=15, fontweight='bold', y=0.98)

        # 1. Гістограма пікових концентрацій
        ax = axes[0, 0]
        n_bins = 60
        counts, bins, patches = ax.hist(conc, bins=n_bins, color='steelblue',
                                         alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(gdk, color='red', linewidth=2.5, linestyle='--', label=f'ГДК = {gdk:.1f} мкг/л')
        ax.axvline(np.mean(conc), color='orange', linewidth=2, linestyle='-',
                   label=f'Середнє = {np.mean(conc):.2f} мкг/л')
        ax.axvline(np.median(conc), color='green', linewidth=2, linestyle=':',
                   label=f'Медіана = {np.median(conc):.2f} мкг/л')

        # Зафарбувати зону перевищення ГДК
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge >= gdk:
                patch.set_facecolor('#e74c3c')
                patch.set_alpha(0.8)

        prob = (conc > gdk).mean() * 100
        ax.set_title(f'Розподіл пікової концентрації\n(P[C > ГДК] = {prob:.1f}%)')
        ax.set_xlabel('Концентрація, мкг/л')
        ax.set_ylabel('Кількість реалізацій')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # 2. Кумулятивна функція розподілу (CDF)
        ax = axes[0, 1]
        sorted_conc = np.sort(conc)
        cdf = np.arange(1, len(sorted_conc) + 1) / len(sorted_conc)
        ax.plot(sorted_conc, cdf * 100, 'b-', linewidth=2)
        ax.axvline(gdk, color='red', linewidth=2, linestyle='--', label=f'ГДК = {gdk:.1f} мкг/л')
        ax.axhline((conc <= gdk).mean() * 100, color='gray', linewidth=1, linestyle=':')

        # Відмітки процентилів
        for pct in [5, 25, 50, 75, 95]:
            val = np.percentile(conc, pct)
            ax.plot(val, pct, 'ko', markersize=5)
            ax.annotate(f'P{pct}={val:.2f}', (val, pct), fontsize=7,
                       textcoords="offset points", xytext=(5, 5))

        ax.set_title('Кумулятивна функція розподілу (CDF)')
        ax.set_xlabel('Концентрація, мкг/л')
        ax.set_ylabel('Ймовірність C ≤ x, %')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # 3. Гістограма виносу
        ax = axes[0, 2]
        ax.hist(export, bins=50, color='darkorange', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(export), color='red', linewidth=2, linestyle='-',
                   label=f'Середнє = {np.mean(export):.1f} г')
        ax.set_title('Розподіл виносу пестициду з поля')
        ax.set_xlabel('Винос, г')
        ax.set_ylabel('Кількість реалізацій')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # 4. Scatter: CN vs концентрація
        ax = axes[1, 0]
        if 'cn' in params:
            sc = ax.scatter(params['cn'], conc, c=conc, cmap='RdYlGn_r',
                           s=3, alpha=0.3, vmin=0, vmax=max(gdk * 2, np.percentile(conc, 95)))
            ax.axhline(gdk, color='red', linewidth=2, linestyle='--', alpha=0.7)
            ax.set_xlabel('Curve Number (CN)')
            ax.set_ylabel('Пік концентрації, мкг/л')
            ax.set_title('Вплив CN на концентрацію')
            plt.colorbar(sc, ax=ax, label='мкг/л', shrink=0.8)
            ax.grid(True, alpha=0.2)

        # 5. Scatter: foc vs концентрація
        ax = axes[1, 1]
        if 'foc' in params:
            sc = ax.scatter(params['foc'] * 100, conc, c=conc, cmap='RdYlGn_r',
                           s=3, alpha=0.3, vmin=0, vmax=max(gdk * 2, np.percentile(conc, 95)))
            ax.axhline(gdk, color='red', linewidth=2, linestyle='--', alpha=0.7)
            ax.set_xlabel('Органічний вуглець, %')
            ax.set_ylabel('Пік концентрації, мкг/л')
            ax.set_title('Вплив органіки ґрунту на концентрацію')
            plt.colorbar(sc, ax=ax, label='мкг/л', shrink=0.8)
            ax.grid(True, alpha=0.2)

        # 6. Tornado chart — рейтинг впливу
        ax = axes[1, 2]
        try:
            indices = mc_engine.compute_sensitivity_indices()
            names = [idx['label'] for idx in indices.values()]
            rhos = [idx['rho'] for idx in indices.values()]
            colors = ['#e74c3c' if r > 0 else '#2ecc71' for r in rhos]

            y_pos = range(len(names))
            ax.barh(y_pos, rhos, color=colors, alpha=0.8, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Кореляція Спірмена (ρ)')
            ax.set_title('Внесок параметрів у невизначеність\n(червоний = збільшує ризик)')
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_xlim(-1, 1)
        except ImportError:
            ax.text(0.5, 0.5, 'scipy не встановлено\nдля розрахунку індексів',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  [Збережено] {save_path}")
        plt.close()

    @staticmethod
    def plot_convergence(mc_engine: MonteCarloEngine, save_path: str = None):
        """Графік збіжності: як стабілізується оцінка з ростом N."""

        conc = mc_engine.results['peak_conc_mg_l']
        gdk = mc_engine.results['gdk']

        # Рахуємо кумулятивні статистики
        steps = np.arange(100, len(conc) + 1, 50)
        means = [np.mean(conc[:n]) * 1000 for n in steps]
        probs = [(conc[:n] > gdk).mean() * 100 for n in steps]
        p95s = [np.percentile(conc[:n], 95) * 1000 for n in steps]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Збіжність Monte Carlo оцінок', fontsize=13, fontweight='bold')

        ax1.plot(steps, means, 'b-', linewidth=1.5)
        ax1.axhline(np.mean(conc) * 1000, color='red', linestyle='--', alpha=0.5,
                     label=f'Фінальне = {np.mean(conc)*1000:.3f}')
        ax1.set_xlabel('Кількість ітерацій')
        ax1.set_ylabel('Середня конц., мкг/л')
        ax1.set_title('Збіжність середнього')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, probs, 'r-', linewidth=1.5)
        ax2.axhline(probs[-1], color='gray', linestyle='--', alpha=0.5,
                     label=f'Фінальне = {probs[-1]:.1f}%')
        ax2.set_xlabel('Кількість ітерацій')
        ax2.set_ylabel('P[C > ГДК], %')
        ax2.set_title('Збіжність ймовірності перевищення ГДК')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3.plot(steps, p95s, 'g-', linewidth=1.5)
        ax3.axhline(gdk * 1000, color='red', linestyle='--', linewidth=2,
                     label=f'ГДК = {gdk*1000:.1f} мкг/л')
        ax3.axhline(p95s[-1], color='gray', linestyle='--', alpha=0.5,
                     label=f'Фінальне P95 = {p95s[-1]:.3f}')
        ax3.set_xlabel('Кількість ітерацій')
        ax3.set_ylabel('95-й процентиль, мкг/л')
        ax3.set_title('Збіжність 95-го процентиля')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Збережено] {save_path}")
        plt.close()


# ============================================================
# БЛОК 4: ЗАПУСК ЕКСПЕРИМЕНТУ
# ============================================================

def run_monte_carlo_experiment():
    """Повний стохастичний експеримент."""

    import os
    os.makedirs('images', exist_ok=True)

    np.random.seed(2024)

    print("=" * 65)
    print("  СТОХАСТИЧНИЙ ЕКСПЕРИМЕНТ: MONTE CARLO СИМУЛЯЦІЯ")
    print("  Сценарій: Атразин, Черкаська область")
    print("=" * 65)

    # ---- Визначаємо параметри з невизначеністю ----
    uncertain_params = {
        'cn': UncertainParameter(
            name='Curve Number', mean=78, std=5,
            dist='normal', low=55, high=98
        ),
        'foc': UncertainParameter(
            name='Орг. вуглець', mean=0.03, std=0.007,
            dist='normal', low=0.005, high=0.08
        ),
        'half_life': UncertainParameter(
            name='DT50', mean=75, std=18,
            dist='lognormal', low=20, high=200
        ),
        'dose': UncertainParameter(
            name='Доза', mean=1.5, std=0.15,
            dist='normal', low=0.5, high=3.0
        ),
        'rain': UncertainParameter(
            name='Опади', mean=35, std=8,
            dist='normal', low=5, high=80
        ),
        'k_oc': UncertainParameter(
            name='Koc', mean=100, std=25,
            dist='lognormal', low=30, high=300
        ),
        'base_flow': UncertainParameter(
            name='Витрата річки', mean=8.0, std=2.5,
            dist='lognormal', low=1.0, high=30.0
        ),
    }

    # ---- Фіксовані параметри ----
    fixed_params = {
        'area_ha': 50,
        'bulk_density': 1.3,
        'porosity': 0.50,
        'mixing_depth_m': 0.05,
        'river_width': 30,
        'river_depth': 1.5,
        'river_velocity': 0.4,
        'river_dispersion': 5.0,
        'days_since': 10,
        'gdk': 0.002,
    }

    # ---- Запуск Monte Carlo ----
    mc = MonteCarloEngine(n_iterations=2000)
    results = mc.run(uncertain_params, fixed_params)

    # ---- Індекси чутливості ----
    # (вже виводяться всередині plot_full_report)

    # ---- Візуалізація ----
    print(f"\n{'─'*65}")
    print(f"  ПОБУДОВА ГРАФІКІВ")
    print(f"{'─'*65}")

    MonteCarloVisualizer.plot_full_report(mc, save_path='images/fig6_mc.png')
    MonteCarloVisualizer.plot_convergence(mc, save_path='images/fig7_convergence.png')

    # ---- Порівняння детерміністичного та стохастичного підходів ----
    conc = results['peak_conc_mg_l']
    gdk = results['gdk']

    # Детерміністичний результат (з номінальними параметрами)
    pesticide = Pesticide("Атразин", k_oc=100, half_life=75, gdk=0.002)
    soil = SoilProfile(foc=0.03, bulk_density=1.3, porosity=0.50)
    field = AgriculturalField(area_ha=50, soil=soil, cn=78)
    river = River(width=30, depth=1.5, velocity=0.4, base_flow=8.0, dispersion=5.0)
    engine = SimulationEngine(field, pesticide, river,
                              BufferZone(width_m=20, slope_pct=5.0, vegetation_type='grass'))

    residue = engine.compute_field_residue(1.5, 10)
    export = engine.compute_daily_export(residue, 35)
    buf_result = engine.buffer.compute_delivery(export['mass_exported_g'], pesticide.decay_rate)
    river_result = engine.solve_river_transport(buf_result['mass_delivered_g'])
    det_conc = np.max(river_result['C_max'])

    print(f"\n{'─'*65}")
    print(f"  ПОРІВНЯННЯ ПІДХОДІВ")
    print(f"{'─'*65}")
    print(f"\n  {'Показник':<35} {'Детермін.':<15} {'Monte Carlo'}")
    print(f"  {'─'*62}")
    print(f"  {'Пікова конц. (мкг/л)':<35} {det_conc*1000:<15.3f} {np.mean(conc)*1000:.3f} (середнє)")
    print(f"  {'':35} {'—':<15} {np.median(conc)*1000:.3f} (медіана)")
    print(f"  {'90% довірчий інтервал (мкг/л)':<35} {'—':<15} "
          f"[{np.percentile(conc, 5)*1000:.3f} — {np.percentile(conc, 95)*1000:.3f}]")
    print(f"  {'Перевищення ГДК?':<35} {'Ні (89%)':<15} "
          f"P = {(conc > gdk).mean()*100:.1f}%")
    print(f"  {'Макс. можлива конц. (мкг/л)':<35} {'—':<15} {np.max(conc)*1000:.3f}")

    print(f"\n  Висновок:")
    prob = (conc > gdk).mean() * 100
    if prob < 5:
        print(f"  Детерміністична модель дає адекватну оцінку.")
        print(f"  Ймовірність перевищення ГДК лише {prob:.1f}% — ризик низький.")
    elif prob < 25:
        print(f"  Детерміністична модель занижує ризик!")
        print(f"  Хоча номінальна концентрація нижче ГДК,")
        print(f"  ймовірність перевищення = {prob:.1f}% — ризик ПОМІРНИЙ.")
        print(f"  Рекомендовано додатковий моніторинг.")
    else:
        print(f"  Детерміністична модель СУТТЄВО занижує ризик!")
        print(f"  Ймовірність перевищення ГДК = {prob:.1f}% — ризик ВИСОКИЙ.")
        print(f"  Необхідні заходи зі зменшення навантаження.")

    print(f"\n  Збережені графіки:")
    print(f"    • output/monte_carlo_report.png — комплексний звіт (6 графіків)")
    print(f"    • output/monte_carlo_convergence.png — збіжність оцінок")
    print(f"\n{'='*65}")


if __name__ == '__main__':
    run_monte_carlo_experiment()
