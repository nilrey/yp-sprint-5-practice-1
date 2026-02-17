import pandas as pd
import numpy as np

path_old = 'music_old.csv'   
path_new = 'music_new.csv' 

print("Загрузка данных...")
# Загружаем данные
df_old = pd.read_csv(path_old)
df_new = pd.read_csv(path_new)

print(f"Размер референсного датасета (old): {df_old.shape}")
print(f"Размер нового датасета (new): {df_new.shape}")

# Убедимся, что колонка 'loudness' существует
if 'loudness' not in df_old.columns or 'loudness' not in df_new.columns:
    print("Ошибка: Колонка 'loudness' не найдена в одном из файлов.")
else:
    # Извлекаем данные по громкости
    old_data = df_old['loudness'].dropna()  # убираем пропуски, если есть
    new_data = df_new['loudness'].dropna()

    # --- Функция для расчета PSI (Population Stability Index) ---
    def calculate_psi(expected, actual, bins=20):
        """
        Рассчитывает PSI для двух распределений.
        expected: референсные данные (old)
        actual: новые данные (new)
        bins: количество бинов для разбивки
        """
        # Определяем границы бинов на основе референсных данных (old)
        # Добавляем +1 к bins, чтобы получить границы, а не метки
        breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))

        # Обработка крайних случаев: если несколько значений равны максимальному перцентилю,
        # границы могут повторяться. Добавим небольшой шум к максимальной границе,
        # чтобы все значения попали в бины.
        breaks[-1] = breaks[-1] + 0.0001

        # Разбиваем данные на бины и считаем частоты (нормированные, в процентах)
        expected_counts = np.histogram(expected, bins=breaks)[0]
        actual_counts = np.histogram(actual, bins=breaks)[0]

        # Преобразуем в проценты (доли)
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)

        # Защита от деления на ноль и логарифма нуля.
        # Если доля = 0, заменяем на очень маленькое число (0.0001), чтобы избежать ошибок.
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        # Расчет PSI по каждому бину и суммирование
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)

        # Для наглядности выведем промежуточную таблицу
        psi_table = pd.DataFrame({
            'Граница_бина_от': breaks[:-1],
            'Граница_бина_до': breaks[1:],
            'Old_%': expected_percents,
            'New_%': actual_percents,
            'PSI_бина': psi_values
        })
        return psi, psi_table

    # --- Расчет PSI для loudness ---
    print("\nРасчет PSI для признака 'loudness'...")
    psi_value, psi_details = calculate_psi(old_data, new_data, bins=20)

    # --- Интерпретация результата ---
    print("\n" + "="*50)
    print(f"ИТОГОВЫЙ PSI = {psi_value:.4f}")
    print("="*50)

    print("\nДетализация по бинам (первые 5 строк):")
    print(psi_details.head())

    # Интерпретация
    print("\nИнтерпретация результата:")
    if psi_value < 0.1:
        print("✅ PSI < 0.1: Дрейфа НЕТ (незначительные изменения).")
    elif psi_value < 0.2:
        print("⚠️ 0.1 <= PSI < 0.2: Требуется внимание (умеренные изменения).")
    else:
        print("❌ PSI >= 0.2: Сильный дрейф данных.")
