# tests/test_data_processor.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.testing import assert_frame_equal, assert_series_equal

# Добавляем путь к родительской директории, чтобы импортировать модули
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_processor import DataProcessor

# Фикстура для создания экземпляра DataProcessor перед каждым тестом
@pytest.fixture
def processor():
    """Возвращает экземпляр DataProcessor."""
    return DataProcessor()

# --- Тестовые данные ---

@pytest.fixture
def sample_data_basic():
    """Простой набор данных для тестов."""
    data = {
        'Материал': [' M1 ', 'M2', 'M1', ' M2 ', 'M3'],
        'ДатаСоздан': ['01.01.2023', '2023-01-15', '10/02/2023', '2023-02-20', '2023-03-01'],
        'Цена нетто': ['100,5', '200', ' 110 ', '210,99', '50'],
        'Курс': ['1', '1.1', '1,0', ' 1,15 ', '1'],
        'Влт': ['RUB', 'USD', 'RUB', 'USD', 'RUB']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_edge_cases():
    """Данные с пропусками, некорректными значениями, отрицательными ценами."""
    data = {
        'Материал': ['M1', 'M2', 'M1', 'M2', 'M3', 'M4', 'M1'],
        'ДатаСоздан': ['01.01.2023', 'invalid-date', '10.02.2023', '20.02.2023', np.nan, '01.03.2023', '05.01.2023'],
        'Цена нетто': ['100', '200', '-50', ' ', '150', np.nan, '100'], # Отрицательная, пустая строка, NaN
        'Курс': ['1', '1.1', '1', 'abc', '-2', np.nan, '1'], # Некорректный, отрицательный, NaN
        'Влт': ['RUB', 'USD', 'RUB', 'EUR', 'RUB', 'RUB', 'RUB']
    }
    return pd.DataFrame(data)

# --- Тесты базовой обработки колонок ---

def test_process_material_column(processor, sample_data_basic):
    """Тестирует обработку колонки 'Материал' (пробелы, тип)."""
    processed_df = processor._process_material_column(sample_data_basic.copy())
    expected_materials = pd.Series(['M1', 'M2', 'M1', 'M2', 'M3'], name='Материал')
    assert_series_equal(processed_df['Материал'], expected_materials, check_dtype=False)

def test_process_material_column_empty(processor):
    """Тестирует обработку пустых значений в 'Материал'."""
    df = pd.DataFrame({'Материал': ['M1', ' ', '', 'M2'], 'Цена нетто': [1, 2, 3, 4]})
    processed_df = processor._process_material_column(df)
    expected_materials = pd.Series(['M1', 'Unknown', 'Unknown', 'M2'], name='Материал')
    assert_series_equal(processed_df['Материал'], expected_materials, check_dtype=False)

def test_process_date_column(processor, sample_data_basic):
    """Тестирует обработку колонки 'ДатаСоздан' (разные форматы)."""
    processed_df = processor._process_date_column(sample_data_basic.copy())
    expected_dates = pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-10', '2023-02-20', '2023-03-01'], errors='coerce')
    assert_series_equal(processed_df['ДатаСоздан'], pd.Series(expected_dates, name='ДатаСоздан'), check_dtype=True)

def test_process_date_column_invalid(processor, sample_data_edge_cases):
    """Тестирует обработку некорректных дат (удаление строк)."""
    df_copy = sample_data_edge_cases.copy()
    # Ожидаем, что строки с 'invalid-date' и NaN будут удалены
    processed_df = processor._process_date_column(df_copy)
    assert processed_df.shape[0] == 5 # 7 - 2 = 5 строк должно остаться
    assert not processed_df['ДатаСоздан'].isna().any()

def test_clean_and_convert_numeric_price(processor, sample_data_basic):
    """Тестирует _clean_and_convert_numeric для 'Цена нетто'."""
    processed_df = processor._clean_and_convert_numeric(sample_data_basic.copy(), 'Цена нетто', allow_negative=False)
    expected_prices = pd.Series([100.5, 200.0, 110.0, 210.99, 50.0], name='Цена нетто')
    assert_series_equal(processed_df['Цена нетто'], expected_prices, check_dtype=True)

def test_clean_and_convert_numeric_price_edge(processor, sample_data_edge_cases):
    """Тестирует _clean_and_convert_numeric для 'Цена нетто' (граничные случаи)."""
    processed_df = processor._clean_and_convert_numeric(sample_data_edge_cases.copy(), 'Цена нетто', allow_negative=False, default_value=np.nan)
    # Ожидаем: 100, 200, NaN (из-за -50), NaN (из-за ' '), 150, NaN, 100
    assert pd.isna(processed_df['Цена нетто'].iloc[2])
    assert pd.isna(processed_df['Цена нетто'].iloc[3])
    assert pd.isna(processed_df['Цена нетто'].iloc[5])
    assert processed_df['Цена нетто'].iloc[0] == 100.0
    assert processed_df['Цена нетто'].iloc[1] == 200.0
    assert processed_df['Цена нетто'].iloc[4] == 150.0
    assert processed_df['Цена нетто'].iloc[6] == 100.0

def test_clean_and_convert_numeric_rate(processor, sample_data_basic):
    """Тестирует _clean_and_convert_numeric для 'Курс'."""
    processed_df = processor._clean_and_convert_numeric(sample_data_basic.copy(), 'Курс', allow_negative=False, default_value=1.0)
    expected_rates = pd.Series([1.0, 1.1, 1.0, 1.15, 1.0], name='Курс')
    assert_series_equal(processed_df['Курс'], expected_rates, check_dtype=True)

def test_clean_and_convert_numeric_rate_edge(processor, sample_data_edge_cases):
    """Тестирует _clean_and_convert_numeric для 'Курс' (граничные случаи)."""
    processed_df = processor._clean_and_convert_numeric(sample_data_edge_cases.copy(), 'Курс', allow_negative=False, default_value=1.0)
    # Ожидаем: 1.0, 1.1, 1.0, 1.0 (из-за 'abc'), 1.0 (из-за -2), 1.0 (из-за NaN), 1.0
    expected_rates = pd.Series([1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0], name='Курс')
    assert_series_equal(processed_df['Курс'], expected_rates, check_dtype=True)

def test_normalize_prices(processor, sample_data_basic):
    """Тестирует нормализацию цен."""
    df = sample_data_basic.copy()
    df = processor._clean_and_convert_numeric(df, 'Цена нетто', default_value=np.nan)
    df = processor._clean_and_convert_numeric(df, 'Курс', default_value=1.0)
    processed_df = processor._normalize_prices(df)
    expected_norm_prices = pd.Series([100.5, 220.0, 110.0, 242.6385, 50.0], name='Цена нетто (норм.)')
    assert 'Цена нетто (норм.)' in processed_df.columns
    assert 'Базовая валюта' in processed_df.columns
    assert_series_equal(processed_df['Цена нетто (норм.)'], expected_norm_prices, check_dtype=True, rtol=1e-5)
    assert processed_df['Базовая валюта'].iloc[0] == 'RUB' # Первая валюта с курсом 1

# --- Тесты добавления признаков ---

def test_add_base_time_features(processor, sample_data_basic):
    """Тестирует добавление базовых временных признаков."""
    df = sample_data_basic.copy()
    df = processor._process_date_column(df) # Нужны корректные даты
    processed_df = processor._add_base_time_features(df)
    expected_cols = ['Год', 'Месяц', 'День', 'Квартал', 'День недели', 'Номер недели', 'День года', 'Дней от начала', 'Сезон']
    for col in expected_cols:
        assert col in processed_df.columns
    assert processed_df['Год'].iloc[0] == 2023
    assert processed_df['Месяц'].iloc[1] == 1
    assert processed_df['День'].iloc[2] == 10
    assert processed_df['Квартал'].iloc[3] == 1
    assert processed_df['День недели'].iloc[0] == 6 # 01.01.2023 - Воскресенье
    assert processed_df['Номер недели'].iloc[0] == 52 # Или 1 в зависимости от года? Проверить ISO
    assert processed_df['День года'].iloc[0] == 1
    assert processed_df['Дней от начала'].iloc[0] == 0
    assert processed_df['Дней от начала'].max() > 0
    assert processed_df['Сезон'].iloc[0] == 'Зима'
    assert processed_df['Сезон'].iloc[4] == 'Весна'


# --- Тесты расчета метрик ---

def test_calculate_and_add_metrics_simple(processor):
    """Тестирует расчет метрик для простого случая (один материал)."""
    data = {
        'Материал': ['M1', 'M1', 'M1'],
        'ДатаСоздан': ['2023-01-01', '2023-01-10', '2023-01-20'],
        'Цена нетто (норм.)': [100.0, 110.0, 100.0]
    }
    df = pd.DataFrame(data)
    df['ДатаСоздан'] = pd.to_datetime(df['ДатаСоздан'])
    
    processed_df = processor._calculate_and_add_metrics(df)
    
    assert 'Количество записей материала' in processed_df.columns
    assert processed_df['Количество записей материала'].iloc[0] == 3
    assert abs(processed_df['Средняя цена материала'].iloc[0] - 103.333) < 0.01
    assert abs(processed_df['Медианная цена материала'].iloc[0] - 100.0) < 0.01
    assert abs(processed_df['Стд. отклонение цены материала'].iloc[0] - 5.7735) < 0.01
    assert abs(processed_df['Коэффициент вариации цены'].iloc[0] - 5.587) < 0.01 # (5.7735 / 103.333) * 100
    assert processed_df['Временной диапазон материала'].iloc[0] == 19 # 20 - 1
    assert abs(processed_df['Процент стабильности цены'].iloc[0] - (2/3)*100) < 0.01 # Цена 100 встречается 2 раза из 3
    assert processed_df['Стабильная цена'].iloc[0] == False # 66.6% < 80%
    # Даты первой/последней
    assert processed_df['Первая дата материала'].iloc[0] == pd.Timestamp('2023-01-01')
    assert processed_df['Последняя дата материала'].iloc[0] == pd.Timestamp('2023-01-20')
    # Дни с последней активности (latest_date = 2023-01-20)
    assert processed_df['Дней с последней активности'].iloc[0] == 0
    assert processed_df['Неактивный материал'].iloc[0] == False

def test_calculate_and_add_metrics_multiple(processor):
    """Тестирует расчет метрик для нескольких материалов."""
    data = {
        'Материал': ['M1', 'M2', 'M1', 'M2', 'M2'],
        'ДатаСоздан': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-25'],
        'Цена нетто (норм.)': [100.0, 200.0, 110.0, 200.0, 200.0]
    }
    df = pd.DataFrame(data)
    df['ДатаСоздан'] = pd.to_datetime(df['ДатаСоздан'])
    processed_df = processor._calculate_and_add_metrics(df)

    # M1
    m1_row = processed_df[processed_df['Материал'] == 'M1'].iloc[0]
    assert m1_row['Количество записей материала'] == 2
    assert m1_row['Средняя цена материала'] == 105.0
    assert m1_row['Временной диапазон материала'] == 9
    assert m1_row['Стабильная цена'] == False # 50%

    # M2
    m2_row = processed_df[processed_df['Материал'] == 'M2'].iloc[0]
    assert m2_row['Количество записей материала'] == 3
    assert m2_row['Средняя цена материала'] == 200.0
    assert m2_row['Стд. отклонение цены материала'] == 0.0
    assert m2_row['Коэффициент вариации цены'] == 0.0
    assert m2_row['Процент стабильности цены'] == 100.0
    assert m2_row['Стабильная цена'] == True # 100%
    assert m2_row['Временной диапазон материала'] == 20
    assert m2_row['Дней с последней активности'] == 0 # latest_date = 2023-01-25

def test_calculate_and_add_metrics_single_record(processor):
    """Тестирует расчет метрик для материала с одной записью."""
    data = {
        'Материал': ['M1'],
        'ДатаСоздан': ['2023-01-01'],
        'Цена нетто (норм.)': [150.0]
    }
    df = pd.DataFrame(data)
    df['ДатаСоздан'] = pd.to_datetime(df['ДатаСоздан'])
    processed_df = processor._calculate_and_add_metrics(df)
    
    assert processed_df['Количество записей материала'].iloc[0] == 1
    assert processed_df['Средняя цена материала'].iloc[0] == 150.0
    assert pd.isna(processed_df['Стд. отклонение цены материала'].iloc[0]) # std для 1 элемента - NaN
    assert processed_df['Коэффициент вариации цены'].iloc[0] == 0.0 # Должно быть 0 после fillna
    assert processed_df['Временной диапазон материала'].iloc[0] == 0
    assert processed_df['Процент стабильности цены'].iloc[0] == 100.0 # Скорректировано для 1 записи
    assert processed_df['Стабильная цена'].iloc[0] == True
    assert processed_df['Дней с последней активности'].iloc[0] == 0

# def test_calculate_and_add_metrics_with_nan(processor):
#     """Тестирует расчет метрик при наличии NaN в ценах/курсах (уже в processed_df)."""
#     # Этот тест сложнее, т.к. NaN должны обрабатываться ДО метрик
#     # Проверим, что агрегация игнорирует NaN корректно
#     data = {
#         'Материал': ['M1', 'M1', 'M1'],
#         'ДатаСоздан': ['2023-01-01', '2023-01-10', '2023-01-20'],
#         'Цена нетто (норм.)': [100.0, np.nan, 110.0]
#     }
#     df = pd.DataFrame(data)
#     df['ДатаСоздан'] = pd.to_datetime(df['ДатаСоздан'])
#     processed_df = processor._calculate_and_add_metrics(df)

#     assert processed_df['Количество записей материала'].iloc[0] == 3 # Считает все строки
#     assert processed_df['Средняя цена материала'].iloc[0] == 105.0 # mean игнорирует NaN
#     assert abs(processed_df['Стд. отклонение цены материала'].iloc[0] - 7.071) < 0.01 # std([100, 110])
#     # Стабильность должна учитывать NaN как отдельное значение
#     # [100, NaN, 110] -> counts: 100: 1, 110: 1, NaN: 1. most_common_count = 1. percentage = (1/3)*100
#     assert abs(processed_df['Процент стабильности цены'].iloc[0] - (1/3)*100) < 0.01
#     assert processed_df['Стабильная цена'].iloc[0] == False

# --- Тесты заполнения пропусков ---

def test_fill_missing_values(processor):
    """Тестирует заполнение пропущенных значений."""
    data = {
        'Материал': ['M1', 'M1', 'M2', 'M2', 'M3'],
        'Числовая': [10.0, np.nan, 20.0, 22.0, np.nan],
        'Категориальная': ['A', 'B', np.nan, 'C', 'A']
    }
    df = pd.DataFrame(data)
    
    # Добавим фейковые метрики, чтобы _fill_missing_values сработал
    df['Цена нетто (норм.)'] = 1 # Пример
    df['ДатаСоздан'] = datetime.now() # Пример
    df = processor._calculate_and_add_metrics(df) # Рассчитаем метрики, чтобы были группы
    
    processed_df = processor._fill_missing_values(df)

    # Числовая: M1 NaN -> median(10) = 10. M3 NaN -> global median(10, 20, 22) = 20
    assert processed_df['Числовая'].iloc[1] == 10.0 
    assert processed_df['Числовая'].iloc[4] == 20.0 
    assert not processed_df['Числовая'].isna().any()

    # Категориальная: M2 NaN -> global mode ('A')
    assert processed_df['Категориальная'].iloc[2] == 'A'
    assert not processed_df['Категориальная'].isna().any()

# --- Тесты всего процесса ---

def test_process_data_end_to_end(processor, sample_data_basic):
    """Тестирует весь метод process_data."""
    processed_df = processor.process_data(sample_data_basic.copy())

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    
    # Проверяем наличие ключевых колонок метрик
    metric_cols = [
        'Материал', 'ДатаСоздан', 'Цена нетто', 'Курс', 'Влт', 
        'Цена нетто (норм.)', 'Базовая валюта', 'Год', 'Месяц', 'День', 
        'Квартал', 'День недели', 'Номер недели', 'Дней от начала', 'Сезон',
        'Количество записей материала', 'Средняя цена материала', 
        'Медианная цена материала', 'Стд. отклонение цены материала', 
        'Коэффициент вариации цены', 'Временной диапазон материала', 
        'Процент стабильности цены', 'Стабильная цена', 'Первая дата материала', 
        'Последняя дата материала', 'Дней с последней активности', 'Неактивный материал'
    ]
    for col in metric_cols:
        assert col in processed_df.columns
        
    # Проверяем типы данных
    assert pd.api.types.is_string_dtype(processed_df['Материал'])
    assert pd.api.types.is_datetime64_any_dtype(processed_df['ДатаСоздан'])
    assert pd.api.types.is_numeric_dtype(processed_df['Цена нетто (норм.)'])
    assert pd.api.types.is_numeric_dtype(processed_df['Количество записей материала'])
    assert pd.api.types.is_numeric_dtype(processed_df['Коэффициент вариации цены'])
    assert pd.api.types.is_bool_dtype(processed_df['Стабильная цена'])
    assert pd.api.types.is_bool_dtype(processed_df['Неактивный материал'])
    
    # Проверяем выборочные значения M1
    m1_rows = processed_df[processed_df['Материал'] == 'M1']
    assert len(m1_rows) == 2
    assert m1_rows['Количество записей материала'].iloc[0] == 2
    assert abs(m1_rows['Средняя цена материала'].iloc[0] - 105.25) < 0.01 # (100.5*1 + 110*1) / 2
    assert m1_rows['Стабильная цена'].iloc[0] == False

def test_process_data_returns_none_on_error(processor):
    """Тестирует, что process_data возвращает None при критической ошибке."""
    # Пример: DataFrame без обязательной колонки 'Материал'
    df_no_material = pd.DataFrame({'ДатаСоздан': ['2023-01-01'], 'Цена нетто': [100]})
    result = processor.process_data(df_no_material)
    assert result is None
    
    # Пример: DataFrame без обязательной колонки 'ДатаСоздан'
    df_no_date = pd.DataFrame({'Материал': ['M1'], 'Цена нетто': [100]})
    result = processor.process_data(df_no_date)
    assert result is None
