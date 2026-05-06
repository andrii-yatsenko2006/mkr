"""МКР з дисципліни «Python for Data Science». Наскрізний кейс «Метеослужба».

Для роботи з базою даних необхідно попередньо запустити Docker-контейнер:
    docker pull asterindex/pfds-mkr-g2-16:latest
    docker run -d -p 3306:3306 --name mkr asterindex/pfds-mkr-g2-16:latest

Скрипт підключається до бази, виконує обробку та зберігає графіки у папку `plots/`.
"""

# ====================================================================
# Прізвище, ім'я, по батькові: Яценко Андрій Васильович
# Група:                       ЗК-32
# Дата виконання:              6 травня 2026
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    """Підключення до БД із циклом очікування підняття MySQL."""
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                raise
            print(f"  Очікування готовності MySQL (спроба {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


# ====================================================================
# БЛОК 1. NumPy
# ====================================================================
def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    # Переведення даних у масиви NumPy
    temp = df_raw['temperature_c'].values
    humidity = df_raw['humidity_pct'].values
    wind = df_raw['wind_speed_ms'].values

    # 1) Обчислення apparent temperature
    apparent = temp - (100 - humidity) / 5
    print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

    # 2) Заміна фізичних викидів на NaN
    temp_clean = np.where((temp > 60) | (temp < -60), np.nan, temp)
    wind_clean = np.where(wind > 100, np.nan, wind)

    temp_outliers = np.sum(np.isnan(temp_clean) & ~np.isnan(temp))
    wind_outliers = np.sum(np.isnan(wind_clean) & ~np.isnan(wind))
    print(f"2) Викидів температури замінено: {temp_outliers}")
    print(f"   Викидів вітру замінено:       {wind_outliers}")

    # 3) Розрахунок статистик (ігноруючи NaN)
    temp_valid = temp_clean[~np.isnan(temp_clean)]
    mean_t = np.nansum(temp_valid) / len(temp_valid)
    median_t = np.nanmedian(temp_valid)
    std_t = np.sqrt(np.nansum((temp_valid - mean_t) ** 2) / len(temp_valid))
    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # 4) Підрахунок морозних та жарких днів
    n_frost = np.nansum(temp_clean < 0)
    n_hot = np.nansum(temp_clean > 30)
    print(f"4) Морозних: {n_frost}    Жарких: {n_hot}")

    # 5) Пошук екстремумів температури
    idx_max = np.nanargmax(temp_clean)
    idx_min = np.nanargmin(temp_clean)

    obs_id_max, dt_max = df_raw.iloc[idx_max][['obs_id', 'datetime']]
    obs_id_min, dt_min = df_raw.iloc[idx_min][['obs_id', 'datetime']]
    print(f"5) Max T: obs_id={obs_id_max} ({dt_max}), Min T: obs_id={obs_id_min} ({dt_min})")


# ====================================================================
# БЛОК 2. Pandas — очищення
# ====================================================================
def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # 1) Огляд типів даних
    print("1) Структура даних до очищення:")
    print(df.info())

    # 2) Форматування дати та встановлення індексу
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 3) Видалення повних дублікатів
    len_before_dup = len(df)
    df.drop_duplicates(inplace=True)
    n_dups = len_before_dup - len(df)
    print(f"2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнення пропусків вологості медіаною по місту/місяцю
    df['month'] = df.index.month
    na_before = df['humidity_pct'].isna().sum()
    df['humidity_pct'] = df.groupby(['city', 'month'])['humidity_pct'].transform(lambda s: s.fillna(s.median()))
    n_filled = na_before - df['humidity_pct'].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Вилучення фізичних викидів
    len_before_outl = len(df)
    mask_temp = df['temperature_c'].between(-60, 60)
    mask_wind = df['wind_speed_ms'].isna() | df['wind_speed_ms'].between(0, 60)
    df = df[mask_temp & mask_wind]
    n_outliers = len_before_outl - len(df)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    # 6) Підсумковий звіт
    print(f"\n   Звіт очищення: {rows_before} → {len(df)} рядків")

    return df


# ====================================================================
# БЛОК 3. Pandas — аналітика
# ====================================================================
def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # 1) Середня температура по містах
    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values()
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах
    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values()
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    # 3) Середньомісячні температури
    resampler = 'ME' if pd.__version__ >= '2.2.0' else 'M'
    monthly_mean = df['temperature_c'].resample(resampler).mean()
    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # 4) Зведена таблиця (місто × місяць)
    pivot = df.pivot_table(index='city', columns='month', values='temperature_c', aggfunc='mean')
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів із сильними опадами (> 5 мм)
    daily_precip = df.groupby(['city', df.index.date])['precipitation_mm'].sum()
    rainy_days = daily_precip[daily_precip > 5].groupby('city').count()
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # 6) Визначення аномального місяця
    df['year'] = df.index.year
    df['norm_temp'] = df.groupby('month')['temperature_c'].transform('mean')
    df['dev'] = df['temperature_c'] - df['norm_temp']

    month_year_dev = df.groupby(['year', 'month'])['dev'].mean()
    anomaly_month = month_year_dev.abs().idxmax()
    anomaly_dev = month_year_dev.loc[anomaly_month]
    print(f"\n6) Аномальний місяць: {anomaly_month}  відхилення = {anomaly_dev:+.2f}°C")

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
    }


# ====================================================================
# БЛОК 4. Matplotlib
# ====================================================================
def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # Графік 1: Місячна динаміка температур для 3 міст
    fig, ax = plt.subplots(figsize=(11, 5))
    top_3_cities = df['city'].unique()[:3]
    for city in top_3_cities:
        city_data = df[df['city'] == city]
        resampler = 'ME' if pd.__version__ >= '2.2.0' else 'M'
        monthly = city_data['temperature_c'].resample(resampler).mean()
        ax.plot(monthly.index, monthly.values, marker='o', label=city)

    ax.set_title("Monthly Mean Temperature Dynamics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 2: Стовпчикова діаграма опадів
    fig, ax = plt.subplots(figsize=(8, 5))
    precip_data = analytics['by_city_precip']
    ax.bar(precip_data.index, precip_data.values, color='skyblue', edgecolor='black')
    ax.set_title("Total Precipitation by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Precipitation (mm)")
    plt.xticks(rotation=45)
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 3: Гістограма розподілу температур
    fig, ax = plt.subplots(figsize=(9, 5))
    temp_valid = df['temperature_c'].dropna()
    ax.hist(temp_valid, bins=40, color='lightgreen', edgecolor='black', alpha=0.8)

    t_mean = temp_valid.mean()
    t_median = temp_valid.median()
    ax.axvline(t_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {t_mean:.1f}')
    ax.axvline(t_median, color='orange', linestyle='dotted', linewidth=2, label=f'Median: {t_median:.1f}')

    ax.set_title("Temperature Distribution")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 4: Теплова карта температур
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = analytics['pivot']
    cax = ax.imshow(pivot.values, cmap='RdYlBu_r', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    fig.colorbar(cax, ax=ax, label='Temperature (°C)')
    ax.set_title("Temperature Heatmap (City vs Month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Графіки успішно збережені у папку {PLOTS_DIR}/")


# ====================================================================

def main() -> None:
    df_raw = load_observations()
    print(f"Успішно завантажено: {df_raw.shape[0]} рядків, {df_raw.shape[1]} колонок")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()

"""
ВИСНОВКИ (5–8 речень).

За результатами аналізу даних можна зробити такі висновки:

* Температурний режим: Найтеплішим містом виявилося Дніпро (середня температура 12.08 °C), тоді як найхолоднішим — Одеса (7.48 °C). Географічно це може пояснюватися специфікою розташування метеостанцій у прибережній зоні та впливом морських циклонів в Одесі.

* Сезонність: Температурна крива має яскраво виражений класичний характер. Пік спеки припадає на червень-липень (до 24.6 °C у Дніпрі), а найнижчі показники стабільно фіксуються в грудні та січні.

* Аномалії: Головною кліматичною аномалією став березень 2023 року. Середньомісячна температура перевищила норму на 3.62 °C, що свідчить про потужну та нетипово ранню хвилю весняного тепла.

* Опади: Найвищий рівень опадів (697.3 мм) та найбільша кількість дощових днів очікувано зафіксовані в Києві, що робить його найвологішим містом серед досліджуваних.

* Практичні рекомендації: 
  1. Варто закладати підвищені бюджети на зимове опалення інфраструктури в хабах з найнижчими середніми температурами (Одеса, Харків).
  2. Під час будівництва складів чи логістичних центрів у київському регіоні критично важливо інвестувати в надійні системи водовідведення та гідроізоляції через часті та інтенсивні опади.

"""