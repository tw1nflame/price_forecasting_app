import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

class MaterialSegmenter:
    """
    Класс для сегментации материалов на основе различных критериев
    """
    
    def __init__(self):
        pass
    
    def analyze_volatility(self, data):
        """
        Анализирует волатильность цен материалов
        """
        st.header("Анализ волатильности цен материалов")
        
        st.write("""
        Волатильность цен показывает, насколько сильно изменяются цены материала 
        со временем. Высокая волатильность может затруднить прогнозирование цен.
        """)
        
        # Вычисляем волатильность для каждого материала
        volatility_data = self._calculate_volatility(data)
        
        # Сохраняем результаты анализа в session_state
        st.session_state.volatility_data = volatility_data
        
        # Отображаем результаты
        st.subheader("Результаты анализа волатильности")
        
        # Статистика по коэффициентам вариации
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Средняя волатильность", 
                f"{volatility_data['Коэффициент вариации'].mean():.2f}%"
            )
        
        with col2:
            st.metric(
                "Медианная волатильность", 
                f"{volatility_data['Коэффициент вариации'].median():.2f}%"
            )
        
        with col3:
            # Процент материалов с низкой волатильностью (менее 10%)
            low_volatility = (volatility_data['Коэффициент вариации'] < 10).sum()
            low_volatility_percent = (low_volatility / len(volatility_data)) * 100
            
            st.metric(
                "Материалы с низкой волатильностью", 
                f"{low_volatility_percent:.1f}%"
            )
        
        # Поиск материала по коду
        search_material = st.text_input("Поиск материала по коду (для анализа волатильности):")
        
        if search_material:
            filtered_materials = volatility_data[volatility_data['Материал'].str.contains(search_material)]
            st.dataframe(filtered_materials, use_container_width=True)
        else:
            # Показываем топ по волатильности
            st.write("Топ-20 материалов с наибольшей волатильностью:")
            st.dataframe(volatility_data.head(20), use_container_width=True)
            
            st.write("Топ-20 материалов с наименьшей ненулевой волатильностью:")
            st.dataframe(
                volatility_data[volatility_data['Коэффициент вариации'] > 0].tail(20), 
                use_container_width=True
            )
    
    def analyze_stability(self, data):
        """
        Анализирует стабильность цен материалов
        """
        st.header("Анализ стабильности цен материалов")
        
        st.write("""
        Стабильные материалы - это материалы, у которых 80% или более записей имеют одинаковую цену.
        Это может указывать на долгосрочные контракты или стабильные рыночные условия.
        """)
        
        # Вычисляем стабильность для каждого материала
        stability_data = self._calculate_stability(data)
        
        # Сохраняем результаты анализа в session_state
        st.session_state.stability_data = stability_data
        
        # Отображаем результаты
        st.subheader("Результаты анализа стабильности")
        
        # Статистика по стабильности
        col1, col2 = st.columns(2)
        
        with col1:
            stable_count = stability_data['Стабильная цена'].sum()
            stable_percent = (stable_count / len(stability_data)) * 100
            
            st.metric(
                "Стабильные материалы", 
                f"{stable_count} ({stable_percent:.1f}%)"
            )
        
        with col2:
            # Средний процент одинаковых значений
            st.metric(
                "Средний % одинаковых значений", 
                f"{stability_data['Процент одинаковых значений'].mean():.2f}%"
            )
        
        # Поиск материала по коду
        search_material = st.text_input("Поиск материала по коду (для анализа стабильности):")
        
        if search_material:
            filtered_materials = stability_data[stability_data['Материал'].str.contains(search_material)]
            st.dataframe(filtered_materials, use_container_width=True)
        else:
            # Показываем материалы с наибольшим процентом одинаковых значений
            st.write("Топ-20 материалов с наибольшим процентом одинаковых значений:")
            st.dataframe(stability_data.head(20), use_container_width=True)
    
    def analyze_inactivity(self, data):
        """
        Анализирует неактивные материалы
        """
        st.header("Анализ неактивных материалов")
        
        st.write("""
        Неактивные материалы - это материалы, по которым не было записей в течение длительного периода времени.
        По умолчанию, материал считается неактивным, если прошло более 365 дней с последней записи.
        """)
        
        # Вычисляем неактивность для каждого материала
        inactivity_data = self._calculate_inactivity(data)
        
        # Сохраняем результаты анализа в session_state
        st.session_state.inactivity_data = inactivity_data
        
        # Отображаем результаты
        st.subheader("Результаты анализа неактивности")
        
        # Статистика по неактивности
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inactive_count = inactivity_data['Неактивный материал'].sum()
            inactive_percent = (inactive_count / len(inactivity_data)) * 100
            
            st.metric(
                "Неактивные материалы", 
                f"{inactive_count} ({inactive_percent:.1f}%)"
            )
        
        with col2:
            # Среднее количество дней с последней активности
            st.metric(
                "Среднее время неактивности", 
                f"{inactivity_data['Дней с последней активности'].mean():.1f} дней"
            )
        
        with col3:
            # Максимальное количество дней с последней активности
            st.metric(
                "Максимальное время неактивности", 
                f"{inactivity_data['Дней с последней активности'].max():.0f} дней"
            )
        
        # Настройка порога неактивности
        inactivity_threshold = st.slider(
            "Порог неактивности (дней)", 
            min_value=30, 
            max_value=1000, 
            value=365, 
            step=30
        )
        
        # Обновляем статус неактивности в соответствии с выбранным порогом
        inactivity_data['Неактивный материал'] = inactivity_data['Дней с последней активности'] > inactivity_threshold
        
        # Вычисляем новые показатели
        inactive_count_new = inactivity_data['Неактивный материал'].sum()
        inactive_percent_new = (inactive_count_new / len(inactivity_data)) * 100
        
        st.info(f"При пороге {inactivity_threshold} дней неактивных материалов: {inactive_count_new} ({inactive_percent_new:.1f}%)")
        
        # Поиск материала по коду
        search_material = st.text_input("Поиск материала по коду (для анализа неактивности):")
        
        if search_material:
            filtered_materials = inactivity_data[inactivity_data['Материал'].str.contains(search_material)]
            st.dataframe(filtered_materials, use_container_width=True)
        else:
            # Показываем материалы с наибольшим периодом неактивности
            st.write("Топ-20 материалов с наибольшим периодом неактивности:")
            st.dataframe(
                inactivity_data.sort_values('Дней с последней активности', ascending=False).head(20), 
                use_container_width=True
            )
    
    def _calculate_volatility(self, data):
        """
        Вычисляет волатильность цен для каждого материала
        """
        # Группируем данные по материалам
        volatility_data = []
        
        for material, group in data.groupby('Материал'):
            # Вычисляем статистики только если есть больше одной записи
            if len(group) > 1:
                # Среднее значение цены
                mean_price = group['Цена нетто'].mean()
                
                # Стандартное отклонение цены
                std_price = group['Цена нетто'].std()
                
                # Коэффициент вариации (в процентах)
                if mean_price != 0:
                    cv = (std_price / mean_price) * 100
                else:
                    cv = 0
                
                # Количество записей
                num_records = len(group)
                
                # Минимальная и максимальная цена
                min_price = group['Цена нетто'].min()
                max_price = group['Цена нетто'].max()
                
                # Диапазон цен
                price_range = max_price - min_price
                
                # Процентное изменение (от минимума к максимуму)
                if min_price != 0:
                    percent_change = (price_range / min_price) * 100
                else:
                    percent_change = 0
                
                volatility_data.append({
                    'Материал': material,
                    'Количество записей': num_records,
                    'Средняя цена': mean_price,
                    'Стандартное отклонение': std_price,
                    'Коэффициент вариации': cv,
                    'Минимальная цена': min_price,
                    'Максимальная цена': max_price,
                    'Диапазон цен': price_range,
                    'Процентное изменение': percent_change
                })
            else:
                # Для материалов с одной записью
                volatility_data.append({
                    'Материал': material,
                    'Количество записей': 1,
                    'Средняя цена': group['Цена нетто'].iloc[0],
                    'Стандартное отклонение': 0,
                    'Коэффициент вариации': 0,
                    'Минимальная цена': group['Цена нетто'].iloc[0],
                    'Максимальная цена': group['Цена нетто'].iloc[0],
                    'Диапазон цен': 0,
                    'Процентное изменение': 0
                })
        
        # Создаем DataFrame и сортируем по коэффициенту вариации
        volatility_df = pd.DataFrame(volatility_data)
        volatility_df = volatility_df.sort_values('Коэффициент вариации', ascending=False)
        
        return volatility_df
    
    def _calculate_stability(self, data):
        """
        Вычисляет стабильность цен для каждого материала
        """
        # Группируем данные по материалам
        stability_data = []
        
        for material, group in data.groupby('Материал'):
            # Вычисляем частоту встречаемости каждого значения цены
            price_counts = group['Цена нетто'].value_counts()
            
            # Количество записей
            num_records = len(group)
            
            # Наиболее часто встречающаяся цена
            most_common_price = price_counts.index[0] if len(price_counts) > 0 else None
            
            # Количество записей с наиболее частой ценой
            most_common_count = price_counts.iloc[0] if len(price_counts) > 0 else 0
            
            # Процент одинаковых значений
            percent_same_value = (most_common_count / num_records) * 100 if num_records > 0 else 0
            
            # Флаг стабильности (80% или более одинаковых значений)
            is_stable = percent_same_value >= 80
            
            # Количество уникальных значений цены
            unique_prices_count = len(price_counts)
            
            stability_data.append({
                'Материал': material,
                'Количество записей': num_records,
                'Наиболее частая цена': most_common_price,
                'Количество одинаковых значений': most_common_count,
                'Процент одинаковых значений': percent_same_value,
                'Стабильная цена': is_stable,
                'Количество уникальных цен': unique_prices_count
            })
        
        # Создаем DataFrame и сортируем по проценту одинаковых значений
        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values('Процент одинаковых значений', ascending=False)
        
        return stability_df
    
    def _calculate_inactivity(self, data):
        """
        Вычисляет неактивность материалов
        """
        # Текущая дата для расчета периода неактивности
        # Используем максимальную дату в данных в качестве "текущей" даты
        current_date = data['ДатаСоздан'].max()
        
        # Группируем данные по материалам
        inactivity_data = []
        
        for material, group in data.groupby('Материал'):
            # Количество записей
            num_records = len(group)
            
            # Первая дата записи
            first_date = group['ДатаСоздан'].min()
            
            # Последняя дата записи
            last_date = group['ДатаСоздан'].max()
            
            # Количество дней между первой и последней записью
            days_span = (last_date - first_date).days if num_records > 1 else 0
            
            # Средний интервал между записями (в днях)
            avg_interval = days_span / (num_records - 1) if num_records > 1 else None
            
            # Количество дней с последней активности
            days_since_last_activity = (current_date - last_date).days
            
            # Флаг неактивности (более 365 дней без записей)
            is_inactive = days_since_last_activity > 365
            
            inactivity_data.append({
                'Материал': material,
                'Количество записей': num_records,
                'Первая дата': first_date,
                'Последняя дата': last_date,
                'Временной диапазон (дни)': days_span,
                'Средний интервал (дни)': avg_interval,
                'Последняя активность материала': last_date,
                'Дней с последней активности': days_since_last_activity,
                'Неактивный материал': is_inactive
            })
        
        # Создаем DataFrame и сортируем по количеству дней с последней активности
        inactivity_df = pd.DataFrame(inactivity_data)
        inactivity_df = inactivity_df.sort_values('Дней с последней активности', ascending=False)
        
        return inactivity_df