import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import get_general_explanation, get_material_specific_explanation, format_streamlit_dataframe

class MaterialSegmenter:
    """
    Класс для анализа и сегментации материалов
    """

    def __init__(self):
        pass

    def analyze_volatility(self, data):
        """
        Анализирует волатильность цен материалов
        """
        st.header("Анализ волатильности цен")
        st.markdown("""
        Волатильность показывает, насколько сильно цена материала изменяется со временем.
        Она измеряется **коэффициентом вариации (CV)**: (Стандартное отклонение цены / Средняя цена) * 100%.
        *   **Низкий CV (< 10-15%):** Цена относительно стабильна.
        *   **Средний CV (15-30%):** Цена умеренно колеблется.
        *   **Высокий CV (> 30-50%):** Цена сильно изменяется, что может указывать на рыночные факторы, ошибки в данных или разные условия закупок.
        """)

        if 'Материал' not in data.columns or 'Цена нетто (норм.)' not in data.columns:
            st.error("Отсутствуют необходимые колонки: 'Материал', 'Цена нетто (норм.)'.")
            return
        if not pd.api.types.is_numeric_dtype(data['Цена нетто (норм.)']):
             st.error("Колонка 'Цена нетто (норм.)' не является числовой.")
             return

        # Рассчитываем волатильность (коэффициент вариации)
        volatility = data.groupby('Материал')['Цена нетто (норм.)'].agg(
            ['count', 'mean', 'std']
        ).reset_index()
        
        # Избегаем деления на ноль и NaN/inf
        volatility['Коэффициент вариации'] = np.where(
            volatility['mean'] != 0, 
            (volatility['std'] / volatility['mean']) * 100, 
            0
        )
        volatility['Коэффициент вариации'] = volatility['Коэффициент вариации'].fillna(0) # Заменяем NaN нулями
        volatility['Коэффициент вариации'] = np.clip(volatility['Коэффициент вариации'], 0, np.inf) # Убираем отрицательные значения, если вдруг появились
        
        volatility.columns = ['Материал', 'Количество записей', 'Средняя цена', 'Стандартное отклонение', 'Коэффициент вариации']
        volatility = volatility.sort_values('Коэффициент вариации', ascending=False)
        
        st.subheader("Таблица волатильности материалов")
        st.dataframe(
            format_streamlit_dataframe(volatility, precision=2),
            use_container_width=True
        )
        # Add general explanation for the volatility table
        st.markdown(get_general_explanation("table_volatility"))

        # Сохраняем результат в сессию для визуализации (записываем лишь если ключ отсутствует)
        if 'volatility_data' not in st.session_state:
            st.session_state.volatility_data = volatility

    def analyze_stability(self, data):
        """
        Анализирует стабильность цен материалов (процент одинаковых значений)
        """
        st.header("Анализ стабильности цен")
        st.markdown("""
        Этот анализ ищет материалы, цена которых почти не меняется.
        Он показывает **процент одинаковых значений цены** - какую долю от всех записей по материалу составляет самая частая цена.
        Высокий процент (например, > 80-90%) указывает на очень стабильную или даже фиксированную цену.
        """)

        if 'Материал' not in data.columns or 'Цена нетто (норм.)' not in data.columns:
            st.error("Отсутствуют необходимые колонки: 'Материал', 'Цена нетто (норм.)'.")
            return

        # Функция для расчета процента наиболее частого значения
        def most_frequent_pct(series):
            if series.empty or series.nunique() == 0:
                return 0
            counts = series.value_counts()
            most_frequent_count = counts.iloc[0]
            return (most_frequent_count / len(series)) * 100

        # Расчет стабильности
        stability = data.groupby('Материал')['Цена нетто (норм.)'].agg(
            ['count', most_frequent_pct]
        ).reset_index()
        
        stability.columns = ['Материал', 'Количество записей', 'Процент одинаковых значений']
        
        # Добавляем флаг стабильности (например, если >= 80%)
        stability['Стабильная цена'] = stability['Процент одинаковых значений'] >= 80
        
        stability = stability.sort_values('Процент одинаковых значений', ascending=False)
        
        st.subheader("Таблица стабильности цен материалов")
        st.dataframe(
            format_streamlit_dataframe(stability, precision=2),
            use_container_width=True
        )
        # Add general explanation for the stability table
        st.markdown(get_general_explanation("table_stability"))

        # Сохраняем результат в сессию для визуализации (записываем лишь если ключ отсутствует)
        if 'stability_data' not in st.session_state:
            st.session_state.stability_data = stability

    def analyze_inactivity(self, data):
        """
        Анализирует неактивные материалы (давно не было записей)
        """
        st.header("Анализ неактивных материалов")
        st.markdown("""
        Этот анализ определяет материалы, по которым давно не было записей (закупок).
        Он рассчитывает количество **дней с последней активности** для каждого материала.
        """)

        if 'Материал' not in data.columns or 'ДатаСоздан' not in data.columns:
            st.error("Отсутствуют необходимые колонки: 'Материал', 'ДатаСоздан'.")
            return
        if not pd.api.types.is_datetime64_any_dtype(data['ДатаСоздан']):
             st.error("Колонка 'ДатаСоздан' не является датой.")
             return

        # Определяем дату последней активности для каждого материала
        last_activity = data.groupby('Материал')['ДатаСоздан'].max().reset_index()
        last_activity.columns = ['Материал', 'Последняя активность материала']
        
        # Определяем текущую (максимальную) дату в данных
        current_date = data['ДатаСоздан'].max()
        
        # Рассчитываем количество дней с последней активности
        last_activity['Дней с последней активности'] = (current_date - last_activity['Последняя активность материала']).dt.days
        
        # Порог неактивности (в днях)
        inactivity_threshold = st.slider(
            "Порог неактивности (дни):", 
            min_value=30, 
            max_value=1095, # 3 года
            value=365, 
            step=30,
            key="inactivity_threshold"
        )
        
        last_activity['Неактивный материал'] = last_activity['Дней с последней активности'] > inactivity_threshold
        
        last_activity = last_activity.sort_values('Дней с последней активности', ascending=False)
        
        st.subheader("Таблица неактивности материалов")
        st.dataframe(
            format_streamlit_dataframe(last_activity, precision=0), # Дни - целые числа
            use_container_width=True
        )
        # Add general explanation for the inactivity table based on the threshold
        st.markdown(get_general_explanation("table_inactivity", threshold=inactivity_threshold))

        # Сохраняем результат в сессии для визуализации (записываем лишь если ключ отсутствует)
        if 'inactivity_data' not in st.session_state:
            st.session_state.inactivity_data = last_activity

    def segment_materials_for_forecast(self, data, min_data_points=24, max_volatility=30, min_activity_days=365, inactivity_threshold=365):
        """
        Сегментирует материалы для выбора метода прогнозирования.
        
        Args:
            data (pd.DataFrame): Обработанные данные.
            min_data_points (int): Минимальное количество записей для ML.
            max_volatility (float): Максимальный CV (%) для ML.
            min_activity_days (int): Минимальный временной диапазон (дни) для ML.
            inactivity_threshold (int): Порог неактивности (дни).
            
        Returns:
            tuple: (dict of DataFrames per segment, dict of segment stats)
        """
        if 'Материал' not in data.columns or 'ДатаСоздан' not in data.columns or 'Цена нетто (норм.)' not in data.columns:
            st.error("Отсутствуют необходимые колонки для сегментации.")
            return {}, {}
        if not pd.api.types.is_datetime64_any_dtype(data['ДатаСоздан']) or not pd.api.types.is_numeric_dtype(data['Цена нетто (норм.)']):
             st.error("Колонки 'ДатаСоздан' или 'Цена нетто (норм.)' имеют неверный формат.")
             return {}, {}

        # 1. Расчет характеристик для каждого материала
        material_stats = data.groupby('Материал').agg(
            count=('ДатаСоздан', 'count'),
            first_date=('ДатаСоздан', 'min'),
            last_date=('ДатаСоздан', 'max'),
            mean_price=('Цена нетто (норм.)', 'mean'),
            std_price=('Цена нетто (норм.)', 'std')
        ).reset_index()

        # 2. Временной диапазон
        material_stats['activity_days'] = (material_stats['last_date'] - material_stats['first_date']).dt.days

        # 3. Коэффициент вариации (CV)
        material_stats['volatility_cv'] = np.where(
            material_stats['mean_price'] != 0,
            (material_stats['std_price'].fillna(0) / material_stats['mean_price']) * 100,
            0
        )
        material_stats['volatility_cv'] = material_stats['volatility_cv'].fillna(0)
        material_stats['volatility_cv'] = np.clip(material_stats['volatility_cv'], 0, np.inf)

        # 4. Дни с последней активности
        current_date = data['ДатаСоздан'].max()
        material_stats['days_since_last_activity'] = (current_date - material_stats['last_date']).dt.days

        # 5. Сегментация
        segments = {
            "ML-прогнозирование": pd.DataFrame(),
            "Наивные методы": pd.DataFrame(),
            "Постоянная цена": pd.DataFrame(),
            "Неактивные": pd.DataFrame(),
            "Недостаточно истории": pd.DataFrame(),
            "Высокая волатильность": pd.DataFrame()
        }

        for index, row in material_stats.iterrows():
            material = row['Материал']
            count = row['count']
            cv = row['volatility_cv']
            activity_days = row['activity_days']
            days_inactive = row['days_since_last_activity']

            # Проверка на неактивность
            if days_inactive > inactivity_threshold:
                segment = "Неактивные"
            # Проверка на постоянную цену (очень низкий CV, но есть данные)
            elif count >= 2 and cv < 1.0:
                 segment = "Постоянная цена"
            # Проверка на недостаточность истории
            elif count < min_data_points:
                segment = "Недостаточно истории"
            # Проверка на высокую волатильность
            elif cv > max_volatility:
                 # Если волатильность высокая, но истории мало, оставляем в 'Недостаточно истории'
                 if activity_days < min_activity_days:
                     segment = "Недостаточно истории" 
                 else:
                     segment = "Высокая волатильность"
            # Проверка на короткий временной диапазон
            elif activity_days < min_activity_days:
                # Если диапазон короткий, но данных много и волатильность низкая -> Наивные
                segment = "Наивные методы"
            # Основное условие для ML
            elif count >= min_data_points and cv <= max_volatility and activity_days >= min_activity_days:
                segment = "ML-прогнозирование"
            # Остальные случаи -> Наивные методы
            else:
                segment = "Наивные методы"
                
            # Добавляем строку в соответствующий сегмент
            segments[segment] = pd.concat([segments[segment], row.to_frame().T], ignore_index=True)

        # 6. Сбор статистики
        stats = {segment: len(df) for segment, df in segments.items()}
        
        # 7. Подготовка данных для экспорта (выбор колонок)
        export_segments = {}
        for segment, df in segments.items():
            if not df.empty:
                export_segments[segment] = df[[
                    'Материал',
                    'count',
                    'first_date',
                    'last_date',
                    'activity_days',
                    'mean_price',
                    'volatility_cv',
                    'days_since_last_activity'
                ]].rename(columns={
                    'count': 'Количество записей материала',
                    'first_date': 'Первая дата активности',
                    'last_date': 'Последняя дата активности',
                    'activity_days': 'Временной диапазон материала',
                    'mean_price': 'Средняя цена материала',
                    'volatility_cv': 'Коэффициент вариации цены',
                    'days_since_last_activity': 'Дней с последней активности'
                })
            else:
                 export_segments[segment] = pd.DataFrame() # Оставляем пустым, если нет данных

        return export_segments, stats