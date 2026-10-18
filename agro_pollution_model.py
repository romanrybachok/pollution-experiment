"""
Математичне та програмне забезпечення моделювання процесу забруднень
ґрунтів та водних ресурсів внаслідок сільськогосподарської діяльності.

Модуль: agro_pollution_model.py
Автор: [Ваше ім'я]
Спеціальність: 121 Інженерія програмного забезпечення

Модель реалізує:
  - Деградацію пестициду (кінетика 1-го порядку)
  - Сорбцію (лінійна ізотерма, Kd = Koc * foc)
  - Гідрологічний стік (метод SCS-CN)
  - Перенос у річці (1D конвекція-дифузія, метод скінченних різниць)
  - Аналіз чутливості (варіація ключових параметрів)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# БЛОК 1: КЛАСИ ПРЕДМЕТНОЇ ОБЛАСТІ
# ============================================================

class Pesticide:
    """Опис пестициду з фізико-хімічними властивостями."""

    def __init__(self, name: str, k_oc: float, half_life: float, gdk: float):
        """
        Args:
            name: Назва пестициду
            k_oc: Коефіцієнт сорбції на органічному вуглеці (л/кг)
            half_life: Період напіврозпаду DT50 (днів)
            gdk: Гранично допустима концентрація у воді (мг/л)
        """
        if half_life <= 0:
            raise ValueError(f"DT50 має бути > 0, отримано: {half_life}")
        if k_oc < 0:
            raise ValueError(f"Koc має бути >= 0, отримано: {k_oc}")

        self.name = name
        self.k_oc = k_oc
        self.half_life = half_life
        self.gdk = gdk
        self.decay_rate = np.log(2) / half_life  # Константа розпаду (1/день)


class SoilProfile:
    """Характеристики ґрунтового профілю."""

    def __init__(self, foc: float, bulk_density: float = 1.4,
                 porosity: float = 0.45, mixing_depth_m: float = 0.05):
        """
        Args:
            foc: Частка органічного вуглецю (0-1)
            bulk_density: Об'ємна густина ґрунту (г/см³)
            porosity: Пористість (0-1)
            mixing_depth_m: Глибина змішування при стоку (м)
        """
        self.foc = foc
        self.bulk_density = bulk_density
        self.porosity = porosity
        self.mixing_depth_m = mixing_depth_m

    def compute_retardation(self, k_oc: float) -> float:
        """Фактор затримки (retardation factor)."""
        kd = k_oc * self.foc
        return 1 + (self.bulk_density / self.porosity) * kd

    def compute_mobile_fraction(self, k_oc: float) -> float:
        """Частка рухомого (розчиненого) пестициду."""
        return 1.0 / self.compute_retardation(k_oc)


class AgriculturalField:
    """Сільськогосподарське поле з гідрологічними параметрами."""

    def __init__(self, area_ha: float, soil: SoilProfile, cn: float):
        """
        Args:
            area_ha: Площа поля (га)
            soil: Профіль ґрунту
            cn: Curve Number за методом SCS (30-100)
        """
        if not 30 <= cn <= 100:
            raise ValueError(f"CN має бути в діапазоні 30-100, отримано: {cn}")

        self.area_ha = area_ha
        self.soil = soil
        self.cn = cn
        self.s_max = 25.4 * (1000.0 / cn - 10)  # Максимальне утримання (мм)

    def calculate_runoff_mm(self, precip_mm: float) -> float:
        """Поверхневий стік за методом SCS-CN (мм)."""
        if precip_mm <= 0:
            return 0.0
        ia = 0.2 * self.s_max  # Початкові абстракції
        if precip_mm > ia:
            return ((precip_mm - ia) ** 2) / (precip_mm + 0.8 * self.s_max)
        return 0.0


class River:
    """Параметри річки-приймача."""

    def __init__(self, width: float = 40.0, depth: float = 2.0,
                 velocity: float = 0.5, base_flow: float = 10.0,
                 dispersion: float = 5.0):
        """
        Args:
            width: Ширина (м)
            depth: Глибина (м)
            velocity: Швидкість течії (м/с)
            base_flow: Базова витрата (м³/с)
            dispersion: Коефіцієнт поздовжньої дисперсії (м²/с)
        """
        self.width = width
        self.depth = depth
        self.velocity = velocity
        self.base_flow = base_flow
        self.dispersion = dispersion


class BufferZone:
    """
    Модель буферної зони між полем і річкою.

    Реалізує два механізми затримки:
    1. Часова затримка (travel_time) — пестицид досягає річки не миттєво
    2. Затухання маси — інфільтрація, сорбція та деградація на шляху

    Ефективність затримки розраховується за емпіричною моделлю
    на основі ширини буферної смуги (Zhang et al., 2010):
        efficiency = 1 - exp(-alpha * width)
    де alpha — параметр ефективності (залежить від типу рослинності).
    """

    def __init__(self, width_m: float = 0.0, slope_pct: float = 5.0,
                 vegetation_type: str = 'grass',
                 infiltration_rate_mm_h: float = 10.0):
        """
        Args:
            width_m: Ширина буферної зони (м). 0 = без буфера (миттєва доставка)
            slope_pct: Ухил схилу (%)
            vegetation_type: Тип рослинності ('grass', 'forest', 'bare')
            infiltration_rate_mm_h: Швидкість інфільтрації ґрунту буфера (мм/год)
        """
        self.width_m = width_m
        self.slope_pct = slope_pct
        self.vegetation_type = vegetation_type
        self.infiltration_rate_mm_h = infiltration_rate_mm_h

        # Параметр ефективності затримки залежно від рослинності
        # (емпіричні значення на основі метааналізу)
        self._alpha = {
            'grass':  0.10,   # трав'яний буфер
            'forest': 0.15,   # лісова смуга (більш ефективна)
            'bare':   0.03,   # оголений ґрунт (мінімальний ефект)
        }.get(vegetation_type, 0.10)

    def compute_delivery(self, mass_input_g: float, pesticide_decay_rate: float = 0.0) -> dict:
        """
        Розрахунок маси пестициду, що проходить через буферну зону.

        Args:
            mass_input_g: Маса, що надходить з поля (г)
            pesticide_decay_rate: Константа деградації пестициду (1/день)

        Returns:
            dict: mass_delivered_g, travel_time_days, retention_efficiency,
                  mass_retained_g
        """
        if self.width_m <= 0:
            # Без буфера — миттєва доставка, без втрат
            return {
                'mass_delivered_g': mass_input_g,
                'travel_time_days': 0.0,
                'retention_efficiency': 0.0,
                'mass_retained_g': 0.0
            }

        # 1. Ефективність затримки (емпірична модель)
        #    Залежить від ширини та типу рослинності
        retention_eff = 1.0 - np.exp(-self._alpha * self.width_m)

        # 2. Корекція на ухил: крутіший схил — менша ефективність
        #    (вода швидше проноситься через буфер)
        slope_factor = max(0.3, 1.0 - 0.02 * self.slope_pct)
        retention_eff *= slope_factor
        retention_eff = min(retention_eff, 0.99)  # не більше 99%

        # 3. Маса після фізичної затримки (інфільтрація + сорбція)
        mass_after_retention = mass_input_g * (1.0 - retention_eff)

        # 4. Час проходження через буфер
        #    Швидкість поверхневого стоку по схилу (спрощено)
        #    v_overland ≈ 0.01-0.1 м/с залежно від ухилу та рослинності
        roughness = {'grass': 0.03, 'forest': 0.02, 'bare': 0.06}.get(
            self.vegetation_type, 0.03)
        v_overland = roughness * (self.slope_pct / 100.0) ** 0.5  # м/с (спрощено)
        v_overland = max(v_overland, 0.005)  # мінімум 0.005 м/с

        travel_time_s = self.width_m / v_overland
        travel_time_days = travel_time_s / 86400.0

        # 5. Додаткова деградація під час транспорту
        if pesticide_decay_rate > 0 and travel_time_days > 0:
            mass_after_decay = mass_after_retention * np.exp(
                -pesticide_decay_rate * travel_time_days)
        else:
            mass_after_decay = mass_after_retention

        mass_retained = mass_input_g - mass_after_decay

        return {
            'mass_delivered_g': max(0.0, mass_after_decay),
            'travel_time_days': travel_time_days,
            'retention_efficiency': retention_eff,
            'mass_retained_g': mass_retained
        }

    def __repr__(self):
        if self.width_m <= 0:
            return "BufferZone(без буфера — миттєва доставка)"
        return (f"BufferZone(width={self.width_m}м, "
                f"vegetation='{self.vegetation_type}', "
                f"slope={self.slope_pct}%)")


# ============================================================
# БЛОК 2: ДВИГУН СИМУЛЯЦІЇ
# ============================================================

class SimulationEngine:
    """Основний двигун моделювання."""

    def __init__(self, field: AgriculturalField, pesticide: Pesticide,
                 river: River, buffer_zone: BufferZone = None):
        self.field = field
        self.pesticide = pesticide
        self.river = river
        self.buffer = buffer_zone if buffer_zone is not None else BufferZone(width_m=0)

    # ---- Поле: деградація + стік ----

    def compute_field_residue(self, initial_dose_kg_ha: float, day: int) -> float:
        """Залишок пестициду на полі на день `day` (кг/га)."""
        return initial_dose_kg_ha * np.exp(-self.pesticide.decay_rate * day)

    def compute_daily_export(self, residue_kg_ha: float, precip_mm: float) -> dict:
        """
        Добовий винос пестициду з поля.

        Returns:
            dict з ключами: runoff_mm, mass_exported_g, mobile_fraction
        """
        runoff_mm = self.field.calculate_runoff_mm(precip_mm)
        f_mobile = self.field.soil.compute_mobile_fraction(self.pesticide.k_oc)

        if runoff_mm > 0 and precip_mm > 0:
            # Маса в шарі змішування (кг)
            mass_in_layer = residue_kg_ha * self.field.area_ha
            # Частка води, що стікає, від загальної кількості опадів
            runoff_fraction = runoff_mm / precip_mm
            # Коефіцієнт екстракції з поверхневого шару (0.01-0.1)
            extraction_coeff = 0.05
            mass_exported_kg = mass_in_layer * f_mobile * runoff_fraction * extraction_coeff
        else:
            mass_exported_kg = 0.0

        return {
            'runoff_mm': runoff_mm,
            'mass_exported_g': mass_exported_kg * 1000,  # Все в грамах
            'mobile_fraction': f_mobile
        }

    # ---- Річка: 1D конвекція-дифузія ----

    def solve_river_transport(self, mass_input_g: float,
                               length_m: float = 2000,
                               duration_s: float = 7200) -> dict:
        """
        Розв'язок 1D рівняння конвекції-дифузії у річці
        методом скінченних різниць (явна схема).

        ∂C/∂t + v·∂C/∂x = D·∂²C/∂x²

        Args:
            mass_input_g: Маса забруднювача, що надходить (г)
            length_m: Довжина ділянки річки (м)
            duration_s: Тривалість моделювання (с)

        Returns:
            dict з результатами: x, C_final, C_max, t_max_at_points
        """
        v = self.river.velocity
        D = self.river.dispersion

        # Просторова та часова сітка
        dx = 10.0   # крок по простору (м)
        nx = int(length_m / dx) + 1

        # Умова стійкості Куранта: dt <= dx/v та dt <= dx²/(2D)
        dt_conv = dx / v if v > 0 else 1e6
        dt_diff = dx**2 / (2 * D) if D > 0 else 1e6
        dt = 0.4 * min(dt_conv, dt_diff)
        nt = int(duration_s / dt) + 1

        x = np.linspace(0, length_m, nx)
        C = np.zeros(nx)

        # Вхідна концентрація: маса надходить протягом 1 години
        # C_source [мг/л] = (масовий_потік [г/с]) / (витрата [л/с])
        # 1 г/л = 1000 мг/л; 1 м³/с = 1000 л/с
        source_duration_s = 3600.0  # 1 година надходження
        mass_rate = mass_input_g / source_duration_s          # г/с
        flow_rate_liters = self.river.base_flow * 1000.0      # л/с
        C_source = (mass_rate / flow_rate_liters) * 1000.0    # мг/л

        C[0] = C_source

        C_max = np.zeros(nx)
        C_history_points = {0: [], nx//4: [], nx//2: [], 3*nx//4: [], nx-1: []}
        t_array = []

        # Часовий цикл (явна схема)
        for step in range(nt):
            C_new = C.copy()
            for i in range(1, nx - 1):
                advection = -v * (C[i] - C[i-1]) / dx
                diffusion = D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
                C_new[i] = C[i] + dt * (advection + diffusion)

            # Гранична умова на вході: джерело діє першу годину
            if step * dt < source_duration_s:
                C_new[0] = C_source
            else:
                C_new[0] = max(0, C[0] + dt * (-v * (C[0]) / dx + D * (C[1] - C[0]) / dx**2))


            C_new[-1] = C_new[-2]  # Вільна гранична умова на виході

            # Обмеження невід'ємності
            C_new = np.maximum(C_new, 0)
            C = C_new
            C_max = np.maximum(C_max, C)

            # Збереження історії для ключових точок
            if step % max(1, nt // 200) == 0:
                t_array.append(step * dt / 60)  # хвилини
                for idx in C_history_points:
                    C_history_points[idx].append(C[idx])

        return {
            'x': x,
            'C_final': C,
            'C_max': C_max,
            'C_history': C_history_points,
            't_minutes': np.array(t_array),
            'dx': dx,
            'dt': dt,
            'nx': nx,
            'nt': nt
        }

    # ---- Повна часова симуляція ----

    def run_temporal_simulation(self, initial_dose_kg_ha: float,
                                 precipitation_series: np.ndarray,
                                 day_of_application: int = 0) -> dict:
        """
        Симуляція по днях за серією опадів.

        Args:
            initial_dose_kg_ha: Доза внесення (кг/га)
            precipitation_series: Масив добових опадів (мм), довжина = кількість днів
            day_of_application: День внесення (0 = перший день)

        Returns:
            dict з часовими рядами
        """
        n_days = len(precipitation_series)

        residue = np.zeros(n_days)
        runoff = np.zeros(n_days)
        export_g = np.zeros(n_days)
        buffer_delivered_g = np.zeros(n_days)
        cumulative_export_g = np.zeros(n_days)
        river_peak_conc = np.zeros(n_days)

        for day in range(n_days):
            days_since = day - day_of_application

            if days_since < 0:
                residue[day] = 0
                continue

            residue[day] = self.compute_field_residue(initial_dose_kg_ha, days_since)

            result = self.compute_daily_export(residue[day], precipitation_series[day])
            runoff[day] = result['runoff_mm']
            export_g[day] = result['mass_exported_g']

            # Пропускаємо через буферну зону
            buf_result = self.buffer.compute_delivery(
                export_g[day], self.pesticide.decay_rate)
            buffer_delivered_g[day] = buf_result['mass_delivered_g']

            cumulative_export_g[day] = cumulative_export_g[max(0, day-1)] + buffer_delivered_g[day]

            # Якщо є винос після буфера — рахуємо пікову концентрацію в річці
            if buffer_delivered_g[day] > 0.01:  # > 0.01 г
                river_result = self.solve_river_transport(buffer_delivered_g[day])
                river_peak_conc[day] = np.max(river_result['C_max'])

        return {
            'days': np.arange(n_days),
            'precipitation': precipitation_series,
            'residue_kg_ha': residue,
            'runoff_mm': runoff,
            'export_g': export_g,
            'buffer_delivered_g': buffer_delivered_g,
            'cumulative_export_g': cumulative_export_g,
            'river_peak_conc_mg_l': river_peak_conc
        }

    # ---- Аналіз чутливості ----

    def sensitivity_analysis(self, initial_dose_kg_ha: float,
                              precip_mm: float, days_since: int,
                              param_name: str, values: np.ndarray) -> dict:
        """
        Аналіз чутливості: як зміна одного параметра впливає на результат.

        Args:
            param_name: 'cn', 'foc', 'half_life', 'dose', 'rain'
            values: Масив значень параметра
        """
        peak_concs = []
        exports = []

        # Зберігаємо оригінальні значення
        orig = {
            'cn': self.field.cn,
            'foc': self.field.soil.foc,
            'half_life': self.pesticide.half_life,
            'dose': initial_dose_kg_ha,
            'rain': precip_mm
        }

        for val in values:
            # Тимчасово змінюємо параметр
            dose = initial_dose_kg_ha
            rain = precip_mm

            if param_name == 'cn':
                self.field.cn = val
                self.field.s_max = 25.4 * (1000.0 / val - 10)
            elif param_name == 'foc':
                self.field.soil.foc = val
            elif param_name == 'half_life':
                self.pesticide.half_life = val
                self.pesticide.decay_rate = np.log(2) / val
            elif param_name == 'dose':
                dose = val
            elif param_name == 'rain':
                rain = val

            residue = self.compute_field_residue(dose, days_since)
            result = self.compute_daily_export(residue, rain)
            exports.append(result['mass_exported_g'])

            # Пропускаємо через буферну зону
            buf_result = self.buffer.compute_delivery(
                result['mass_exported_g'], self.pesticide.decay_rate)
            mass_to_river = buf_result['mass_delivered_g']

            if mass_to_river > 0.01:
                river = self.solve_river_transport(mass_to_river)
                peak_concs.append(np.max(river['C_max']))
            else:
                peak_concs.append(0)

        # Відновлюємо оригінальні значення
        self.field.cn = orig['cn']
        self.field.s_max = 25.4 * (1000.0 / orig['cn'] - 10)
        self.field.soil.foc = orig['foc']
        self.pesticide.half_life = orig['half_life']
        self.pesticide.decay_rate = np.log(2) / orig['half_life']

        return {
            'param_name': param_name,
            'values': values,
            'exports_g': np.array(exports),
            'peak_conc_mg_l': np.array(peak_concs)
        }


# ============================================================
# БЛОК 3: ВІЗУАЛІЗАЦІЯ
# ============================================================

class Visualizer:
    """Візуалізація результатів моделювання."""

    @staticmethod
    def plot_temporal_simulation(sim_results: dict, pesticide: Pesticide,
                                  save_path: str = None):
        """4 графіки: опади, залишок, винос, концентрація в річці."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f'Часова симуляція: {pesticide.name}', fontsize=14, fontweight='bold')
        days = sim_results['days']

        # 1. Опади
        axes[0].bar(days, sim_results['precipitation'], color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Опади, мм')
        axes[0].set_title('Добові опади')
        axes[0].invert_yaxis()

        # 2. Залишок на полі
        axes[1].plot(days, sim_results['residue_kg_ha'], 'g-', linewidth=2)
        axes[1].set_ylabel('Залишок, кг/га')
        axes[1].set_title('Деградація пестициду на полі')
        axes[1].fill_between(days, sim_results['residue_kg_ha'], alpha=0.2, color='green')

        # 3. Винос з поля
        ax3a = axes[2]
        ax3b = ax3a.twinx()
        ax3a.bar(days, sim_results['export_g'], color='orange', alpha=0.7, label='Добовий винос')
        ax3b.plot(days, sim_results['cumulative_export_g'], 'r-', linewidth=2, label='Накопичений')
        ax3a.set_ylabel('Добовий винос, г')
        ax3b.set_ylabel('Накопичений винос, г', color='red')
        ax3a.set_title('Винос пестициду з поля')
        lines1, labels1 = ax3a.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3a.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 4. Концентрація в річці vs ГДК
        axes[3].bar(days, sim_results['river_peak_conc_mg_l'], color='darkred', alpha=0.7)
        axes[3].axhline(y=pesticide.gdk, color='red', linestyle='--', linewidth=2,
                        label=f'ГДК = {pesticide.gdk} мг/л')
        axes[3].set_ylabel('Пік конц., мг/л')
        axes[3].set_xlabel('День')
        axes[3].set_title('Пікова концентрація в річці')
        axes[3].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Збережено] {save_path}")
        plt.close()

    @staticmethod
    def plot_river_plume(river_result: dict, gdk: float, save_path: str = None):
        """Профіль концентрації вздовж річки + часова динаміка."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Перенос забруднювача в річці (1D конвекція-дифузія)',
                     fontsize=13, fontweight='bold')

        # Ліворуч: максимальна концентрація вздовж річки
        ax1.plot(river_result['x'], river_result['C_max'] * 1000, 'b-', linewidth=2,
                 label='Макс. концентрація')
        ax1.plot(river_result['x'], river_result['C_final'] * 1000, 'g--', linewidth=1.5,
                 label='Кінцева концентрація')
        ax1.axhline(y=gdk * 1000, color='red', linestyle='--', label=f'ГДК = {gdk*1000:.1f} мкг/л')
        ax1.set_xlabel('Відстань від джерела, м')
        ax1.set_ylabel('Концентрація, мкг/л')
        ax1.set_title('Просторовий профіль')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Праворуч: часова динаміка в ключових точках
        t = river_result['t_minutes']
        nx = river_result['nx']
        points = {0: 'Вхід (0 м)', nx//4: f'{river_result["x"][nx//4]:.0f} м',
                  nx//2: f'{river_result["x"][nx//2]:.0f} м',
                  nx-1: f'{river_result["x"][-1]:.0f} м'}
        colors = ['red', 'orange', 'blue', 'green']

        for (idx, label), color in zip(points.items(), colors):
            data = river_result['C_history'][idx]
            if len(data) == len(t):
                ax2.plot(t, np.array(data) * 1000, color=color, linewidth=1.5, label=label)

        ax2.axhline(y=gdk * 1000, color='red', linestyle='--', alpha=0.5,
                     label=f'ГДК = {gdk*1000:.1f} мкг/л')
        ax2.set_xlabel('Час, хв')
        ax2.set_ylabel('Концентрація, мкг/л')
        ax2.set_title('Часова динаміка')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Збережено] {save_path}")
        plt.close()

    @staticmethod
    def plot_sensitivity(results_list: list, gdk: float, save_path: str = None):
        """Графіки аналізу чутливості для кількох параметрів."""
        n = len(results_list)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        fig.suptitle('Аналіз чутливості: пікова концентрація в річці',
                     fontsize=13, fontweight='bold')

        labels_ua = {
            'cn': 'Curve Number (CN)',
            'foc': 'Частка орг. вуглецю (foc)',
            'half_life': 'DT50 (днів)',
            'dose': 'Доза внесення (кг/га)',
            'rain': 'Кількість опадів (мм)'
        }

        for ax, res in zip(axes, results_list):
            ax.plot(res['values'], res['peak_conc_mg_l'] * 1000, 'bo-', linewidth=2,
                    markersize=6)
            ax.axhline(y=gdk * 1000, color='red', linestyle='--', alpha=0.7,
                       label=f'ГДК = {gdk*1000:.1f} мкг/л')
            ax.set_xlabel(labels_ua.get(res['param_name'], res['param_name']))
            ax.set_ylabel('Пік конц., мкг/л')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Збережено] {save_path}")
        plt.close()


# ============================================================
# БЛОК 4: СЦЕНАРІЙ — АТРАЗІН, ЧЕРКАСЬКА ОБЛАСТЬ
# ============================================================

def run_experiment():
    """Повний експеримент з результатами."""

    print("=" * 65)
    print("  МОДЕЛЮВАННЯ ЗАБРУДНЕННЯ ВОДНИХ РЕСУРСІВ АТРАЗІНОМ")
    print("  Сценарій: поле кукурудзи, Черкаська область")
    print("=" * 65)

    # ---- 1. Вхідні дані ----
    print("\n[1] ВХІДНІ ПАРАМЕТРИ")

    atrazine = Pesticide(
        name="Атразин",
        k_oc=100,       # Koc = 100 л/кг (помірна сорбція)
        half_life=75,    # DT50 = 75 днів (стійкий)
        gdk=0.002       # MAC-EQS за Директивою 2013/39/EU для поверхневих вод (мг/л)
    )

    soil = SoilProfile(
        foc=0.03,              # 3% органічного вуглецю (типовий чорнозем)
        bulk_density=1.3,      # г/см³
        porosity=0.50,         # Пористість чорнозему
        mixing_depth_m=0.05    # 5 см поверхневий шар
    )

    field = AgriculturalField(area_ha=50, soil=soil, cn=78)

    river = River(
        width=30,          # м
        depth=1.5,         # м
        velocity=0.4,      # м/с
        base_flow=8.0,     # м³/с
        dispersion=5.0     # м²/с
    )

    print(f"  Пестицид:    {atrazine.name} (Koc={atrazine.k_oc}, DT50={atrazine.half_life} днів)")
    print(f"  ГДК:         {atrazine.gdk} мг/л ({atrazine.gdk*1000} мкг/л)")
    print(f"  Ґрунт:       чорнозем, foc={soil.foc}, ρb={soil.bulk_density} г/см³")
    print(f"  Поле:        {field.area_ha} га, CN={field.cn}")
    print(f"  Річка:       {river.width}×{river.depth} м, v={river.velocity} м/с, Q={river.base_flow} м³/с")

    R = soil.compute_retardation(atrazine.k_oc)
    f_mob = soil.compute_mobile_fraction(atrazine.k_oc)
    print(f"\n  Розрахункові:")
    print(f"    Kd = Koc × foc = {atrazine.k_oc} × {soil.foc} = {atrazine.k_oc * soil.foc:.1f} л/кг")
    print(f"    Фактор затримки R = {R:.2f}")
    print(f"    Рухома фракція = {f_mob:.4f} ({f_mob*100:.2f}%)")

    # ---- 2. Одноразова подія (злива) ----
    print("\n" + "-" * 65)
    print("[2] ОДНОРАЗОВА ПОДІЯ: злива 35 мм через 10 днів після внесення")
    print("-" * 65)

    buffer = BufferZone(width_m=20, slope_pct=5.0, vegetation_type='grass')
    engine = SimulationEngine(field, atrazine, river, buffer)

    print(f"\n  Буферна зона: {buffer}")

    dose = 1.5  # кг/га
    days_since = 10
    rain = 35  # мм

    residue = engine.compute_field_residue(dose, days_since)
    export = engine.compute_daily_export(residue, rain)

    # Пропускаємо через буфер
    buf_result = buffer.compute_delivery(export['mass_exported_g'], atrazine.decay_rate)
    mass_to_river = buf_result['mass_delivered_g']

    print(f"\n  Доза внесення:     {dose} кг/га ({dose * field.area_ha} кг на все поле)")
    print(f"  Залишок (день {days_since}): {residue:.4f} кг/га ({residue/dose*100:.1f}% від початкової)")
    print(f"  Стік (SCS-CN):     {export['runoff_mm']:.2f} мм")
    print(f"  Винос з поля:      {export['mass_exported_g']:.2f} г")
    print(f"  Затримка буфером:  {buf_result['retention_efficiency']*100:.1f}%")
    print(f"  Доставлено в річку:{mass_to_river:.2f} г")

    # Річковий транспорт (з масою після буфера)
    river_result = engine.solve_river_transport(mass_to_river,
                                                 length_m=2000, duration_s=7200)
    peak_conc = np.max(river_result['C_max'])

    print(f"\n  Річковий транспорт (2 км, 2 год):")
    print(f"    Сітка: {river_result['nx']} точок × {river_result['nt']} кроків")
    print(f"    dx = {river_result['dx']} м, dt = {river_result['dt']:.3f} с")
    print(f"    Пікова концентрація: {peak_conc:.6f} мг/л ({peak_conc*1000:.3f} мкг/л)")

    ratio = peak_conc / atrazine.gdk
    if ratio > 1:
        print(f"\n  ⚠️  ПЕРЕВИЩЕННЯ ГДК у {ratio:.1f} разів!")
    else:
        print(f"\n  ✅  Концентрація = {ratio*100:.1f}% від ГДК — в межах норми")

    # Зберігаємо графік річки
    Visualizer.plot_river_plume(river_result, atrazine.gdk,
                                save_path='images/fig2_profile.png')

    # ---- 3. Часова симуляція (90 днів) ----
    print("\n" + "-" * 65)
    print("[3] ЧАСОВА СИМУЛЯЦІЯ: 90 днів з реалістичними опадами")
    print("-" * 65)

    # Генеруємо реалістичну серію опадів для літа Черкаської області
    np.random.seed(42)
    n_days = 90

    # Базова модель: більшість днів без дощу, зрідка зливи
    precip = np.zeros(n_days)
    rainy_days = np.random.choice(n_days, size=18, replace=False)  # ~20% дощових днів
    for d in rainy_days:
        precip[d] = np.random.exponential(scale=12)  # Середні опади ~12 мм

    # Додаємо кілька сильних злив
    precip[15] = 35   # Сильна злива на 15-й день
    precip[32] = 42   # Ще сильніша на 32-й
    precip[58] = 28   # Помірна на 58-й

    day_of_application = 5  # Внесення на 5-й день

    print(f"  Період: {n_days} днів")
    print(f"  Внесення: день {day_of_application}, доза = {dose} кг/га")
    print(f"  Загальні опади: {precip.sum():.0f} мм за {(precip > 0).sum()} дощових днів")
    print(f"  Макс. добовий дощ: {precip.max():.0f} мм")

    sim = engine.run_temporal_simulation(dose, precip, day_of_application)

    # Статистика
    total_export = sim['cumulative_export_g'][-1]
    total_applied = dose * field.area_ha * 1000  # г
    days_above_gdk = np.sum(sim['river_peak_conc_mg_l'] > atrazine.gdk)
    max_river_conc = np.max(sim['river_peak_conc_mg_l'])

    print(f"\n  Результати:")
    print(f"    Загальний винос: {total_export:.1f} г ({total_export/total_applied*100:.3f}% від внесеного)")
    print(f"    Макс. пік у річці: {max_river_conc:.6f} мг/л ({max_river_conc*1000:.3f} мкг/л)")
    print(f"    Днів з перевищенням ГДК: {days_above_gdk} з {n_days}")

    Visualizer.plot_temporal_simulation(sim, atrazine, save_path='images/fig3_dynamics.png')

    # ---- 4. Аналіз чутливості ----
    print("\n" + "-" * 65)
    print("[4] АНАЛІЗ ЧУТЛИВОСТІ")
    print("-" * 65)

    sens_cn = engine.sensitivity_analysis(dose, 35, 10, 'cn',
                                           np.arange(60, 96, 5))
    sens_foc = engine.sensitivity_analysis(dose, 35, 10, 'foc',
                                            np.arange(0.005, 0.06, 0.005))
    sens_rain = engine.sensitivity_analysis(dose, 35, 10, 'rain',
                                             np.arange(10, 65, 5))
    sens_dose = engine.sensitivity_analysis(dose, 35, 10, 'dose',
                                             np.arange(0.5, 3.5, 0.25))

    print("  Параметр           | Діапазон          | Макс. конц. (мкг/л)")
    print("  " + "-" * 60)
    for s in [sens_cn, sens_foc, sens_rain, sens_dose]:
        name = s['param_name'].ljust(18)
        rng = f"{s['values'][0]:.3f} - {s['values'][-1]:.3f}".ljust(18)
        mc = f"{np.max(s['peak_conc_mg_l'])*1000:.3f}"
        print(f"  {name} | {rng} | {mc}")

    Visualizer.plot_sensitivity([sens_cn, sens_foc, sens_rain, sens_dose],
                                 atrazine.gdk, save_path='images/fig4_sensitivity.png')

    # ---- 5. Порівняння пестицидів ----
    print("\n" + "-" * 65)
    print("[5] ПОРІВНЯННЯ ПЕСТИЦИДІВ")
    print("-" * 65)

    # MAC-EQS за Директивою 2013/39/EU для внутрішніх поверхневих вод (мкг/л → мг/л)
    # Koc, DT50 та рекомендовані дози за PPDB (Lewis et al., 2016)
    pesticides_data = [
        {"pest": Pesticide("Атразин",     k_oc=100, half_life=75, gdk=0.002),   "mac_eqs_ugl": 2.0, "dose": 1.5},
        {"pest": Pesticide("Алахлор",     k_oc=124, half_life=14, gdk=0.0007),  "mac_eqs_ugl": 0.7, "dose": 2.5},
        {"pest": Pesticide("Ізопротурон", k_oc=122, half_life=23, gdk=0.001),   "mac_eqs_ugl": 1.0, "dose": 1.5},
        {"pest": Pesticide("Симазин",     k_oc=130, half_life=60, gdk=0.004),   "mac_eqs_ugl": 4.0, "dose": 1.8},
    ]

    print(f"  {'Пестицид':<14} {'Koc':<6} {'DT50':<6} {'Доза':<6} {'MAC-EQS':<10} {'Винос г':<10} {'Пік мкг/л':<12} {'Пік/MAC %'}")
    print("  " + "-" * 80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Порівняння пестицидів Директиви 2013/39/EU', fontsize=13, fontweight='bold')

    names, exports_vals, peaks_vals, mac_pcts = [], [], [], []

    for pd in pesticides_data:
        p = pd["pest"]
        mac_eqs = pd["mac_eqs_ugl"]
        d = pd["dose"]

        eng = SimulationEngine(field, p, river, buffer)
        res = eng.compute_field_residue(d, 10)
        exp = eng.compute_daily_export(res, 35)

        buf_r = buffer.compute_delivery(exp['mass_exported_g'], p.decay_rate)
        mass_delivered = buf_r['mass_delivered_g']

        if mass_delivered > 0.01:
            riv = eng.solve_river_transport(mass_delivered)
            peak = np.max(riv['C_max'])
        else:
            peak = 0

        peak_ugl = peak * 1000
        pct = (peak_ugl / mac_eqs) * 100 if mac_eqs > 0 else 0
        status = "⚠️ ПЕРЕВИЩ." if pct > 100 else "✅ Норма"

        print(f"  {p.name:<14} {p.k_oc:<6} {p.half_life:<6} {mac_eqs:<10} "
              f"{exp['mass_exported_g']:<10.2f} {peak_ugl:<12.4f} {pct:<8.1f}% {status}")

        names.append(p.name)
        exports_vals.append(exp['mass_exported_g'])
        peaks_vals.append(peak_ugl)
        mac_pcts.append(pct)

    colors = ['#e74c3c' if p > 100 else '#2ecc71' for p in mac_pcts]

    ax1.barh(names, exports_vals, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Винос з поля, г')
    ax1.set_title('Маса виносу')
    ax1.grid(True, alpha=0.3, axis='x')

    ax2.barh(names, mac_pcts, color=colors, alpha=0.8)
    ax2.axvline(x=100, color='red', linestyle='--', linewidth=2, label='MAC-EQS = 100%')
    ax2.set_xlabel('Пік / MAC-EQS, %')
    ax2.set_title('Ризик перевищення MAC-EQS')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('images/fig5_compare.png', dpi=150, bbox_inches='tight')
    print(f"\n  [Збережено] images/fig5_compare.png")
    plt.close()

    # ---- 6. Вплив буферної зони ----
    print("\n" + "-" * 65)
    print("[6] ВПЛИВ БУФЕРНОЇ ЗОНИ")
    print("-" * 65)

    buffer_widths = [0, 5, 10, 20, 30, 50]
    veg_types = ['grass', 'forest']

    print(f"\n  Умови: злива 35 мм, день 10, ухил 5%")
    print(f"\n  {'Буфер, м':<12} {'Тип':<10} {'Ефект., %':<12} {'Доставлено, г':<16} {'Пік, мкг/л':<14} {'Пік/ГДК'}")
    print("  " + "-" * 80)

    # Дані для графіка
    results_buffer = {}

    for veg in veg_types:
        results_buffer[veg] = {'widths': [], 'peaks': [], 'efficiencies': [], 'ratios': []}

        for bw in buffer_widths:
            buf = BufferZone(width_m=bw, slope_pct=5.0, vegetation_type=veg)
            buf_result = buf.compute_delivery(
                export['mass_exported_g'], atrazine.decay_rate)

            mass_to_river = buf_result['mass_delivered_g']

            if mass_to_river > 0.01:
                eng_buf = SimulationEngine(field, atrazine, river, buf)
                riv_buf = eng_buf.solve_river_transport(mass_to_river)
                peak_buf = np.max(riv_buf['C_max'])
            else:
                peak_buf = 0

            ratio_buf = peak_buf / atrazine.gdk if atrazine.gdk > 0 else 0
            eff = buf_result['retention_efficiency'] * 100

            results_buffer[veg]['widths'].append(bw)
            results_buffer[veg]['peaks'].append(peak_buf * 1000)
            results_buffer[veg]['efficiencies'].append(eff)
            results_buffer[veg]['ratios'].append(ratio_buf)

            status = "⚠️ ПЕРЕВИЩ." if ratio_buf > 1 else "✅ Норма"
            veg_ua = {'grass': 'трава', 'forest': 'ліс'}[veg]
            print(f"  {bw:<12} {veg_ua:<10} {eff:<12.1f} {mass_to_river:<16.2f} "
                  f"{peak_buf*1000:<14.3f} {ratio_buf:<8.1f} {status}")

    # Графік впливу буферної зони
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Вплив буферної зони на концентрацію пестициду в річці',
                 fontsize=13, fontweight='bold')

    colors_veg = {'grass': '#2ecc71', 'forest': '#27ae60'}
    labels_veg = {'grass': 'Трав\'яний буфер', 'forest': 'Лісова смуга'}

    for veg in veg_types:
        r = results_buffer[veg]
        ax1.plot(r['widths'], r['peaks'], 'o-', color=colors_veg[veg],
                 linewidth=2, markersize=8, label=labels_veg[veg])

    ax1.axhline(y=atrazine.gdk * 1000, color='red', linestyle='--', linewidth=2,
                label=f'ГДК = {atrazine.gdk*1000:.1f} мкг/л')
    ax1.set_xlabel('Ширина буферної зони, м')
    ax1.set_ylabel('Пікова концентрація, мкг/л')
    ax1.set_title('Пікова концентрація vs ширина буфера')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    for veg in veg_types:
        r = results_buffer[veg]
        ax2.plot(r['widths'], r['efficiencies'], 's-', color=colors_veg[veg],
                 linewidth=2, markersize=8, label=labels_veg[veg])

    ax2.set_xlabel('Ширина буферної зони, м')
    ax2.set_ylabel('Ефективність затримки, %')
    ax2.set_title('Ефективність буферної зони')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('images/fig8_buffer.png', dpi=150, bbox_inches='tight')
    print(f"\n  [Збережено] images/fig8_buffer.png")
    plt.close()

    # ---- Підсумок ----
    print("\n" + "=" * 65)
    print("  ПІДСУМОК")
    print("=" * 65)
    print(f"""
  Модель реалізує повний ланцюжок:
    Внесення → Деградація → Сорбція → Стік → Буферна зона → Річковий транспорт

  Для сценарію «Атразин + Черкаська область»:
    • Через 10 днів після внесення залишається {residue/dose*100:.0f}% пестициду
    • При зливі 35 мм з поля виноситься {export['mass_exported_g']:.1f} г атразину
    • За 90-денну симуляцію загальний винос = {total_export:.0f} г 
      ({total_export/total_applied*100:.3f}% від внесеного)
    • Днів з перевищенням ГДК у річці: {days_above_gdk}
    • Буферна зона 30 м (трава) зменшує пік на {results_buffer['grass']['efficiencies'][4]:.0f}%

  Збережені графіки:
    • images/fig2_profile.png — профіль шлейфу в річці
    • images/fig3_dynamics.png — часова динаміка (90 днів)
    • images/fig4_sensitivity.png — аналіз чутливості
    • images/fig5_compare.png — порівняння пестицидів
    • images/fig8_buffer.png — вплив буферної зони
""")


if __name__ == '__main__':
    import os
    os.makedirs('images', exist_ok=True)
    run_experiment()
