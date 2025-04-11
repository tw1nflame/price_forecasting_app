# filepath: c:\price_forecasting_app\modules\data_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.utils import format_streamlit_dataframe, get_general_explanation, get_material_specific_explanation

# --- Helper Functions for Explanations ---

# --- End Helper Functions ---

class DataAnalyzer:
    """
    Класс для анализа данных о материалах
    """
    
    def __init__(self):
        pass
    
    def render_overview(self, data):
        """
        Отображает общий обзор данных
        """
        st.header("Общий обзор данных")
        
        # Основные статистики
        st.subheader("Основные статистики")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Количество записей", f"{data.shape[0]:,}".replace(",", " "))
        
        with col2:
            unique_materials = data["Материал"].nunique()
            st.metric("Уникальных материалов", f"{unique_materials:,}".replace(",", " "))
        
        with col3:
            if pd.api.types.is_datetime64_any_dtype(data['ДатаСоздан']):
                date_range = (data["ДатаСоздан"].max() - data["ДатаСоздан"].min()).days
                st.metric("Временной диапазон", f"{date_range} дней")
            else:
                st.metric("Временной диапазон", "N/A")
        
        # Информация о временном диапазоне
        st.subheader("Временной диапазон")
        
        col1, col2 = st.columns(2)
        if pd.api.types.is_datetime64_any_dtype(data['ДатаСоздан']):
            with col1:
                st.write(f"Начальная дата: {data['ДатаСоздан'].min().strftime('%d.%m.%Y')}")
            with col2:
                st.write(f"Конечная дата: {data['ДатаСоздан'].max().strftime('%d.%m.%Y')}")
        else:
             st.warning("Столбец 'ДатаСоздан' не является датой.")
        
        # Распределение по годам
        st.subheader("Распределение записей по годам")
        
        if 'Год' in data.columns:
            years_count = data.groupby("Год")["Материал"].count().reset_index()
            years_count.columns = ["Год", "Количество записей"]
            
            fig = px.bar(
                years_count, 
                x="Год", 
                y="Количество записей",
                text="Количество записей",
                title="Количество записей по годам"
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("bar_years"))
        else:
            st.warning("Столбец 'Год' отсутствует. Невозможно построить график.")
        
        # Валюты
        if "Влт" in data.columns:
            st.subheader("Распределение по валютам")
            
            currency_count = data.groupby("Влт")["Материал"].count().reset_index()
            currency_count.columns = ["Валюта", "Количество записей"]
            currency_count = currency_count.sort_values("Количество записей", ascending=False)
            
            fig = px.pie(
                currency_count, 
                names="Валюта", 
                values="Количество записей",
                title="Распределение записей по валютам"
            )
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("pie_currency"))
        else:
            st.info("Столбец 'Влт' (Валюта) отсутствует в данных.")
        
        # Распределение цен
        st.subheader("Распределение цен")
        
        if 'Цена нетто' in data.columns and pd.api.types.is_numeric_dtype(data['Цена нетто']):
            # Вычисляем квантили для определения границ гистограммы
            q_low = data["Цена нетто"].quantile(0.01)
            q_high = data["Цена нетто"].quantile(0.99)
            # Убедимся, что границы разумны
            if q_low >= q_high:
                 q_low = data["Цена нетто"].min()
                 q_high = data["Цена нетто"].max()

            fig = px.histogram(
                data[data["Цена нетто"].between(q_low, q_high)], # Фильтруем выбросы здесь
                x="Цена нетто",
                nbins=50,
                # range_x=[q_low, q_high], # Убираем range_x, так как данные уже отфильтрованы
                title="Распределение цен (1% и 99% перцентили)"
            )
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("hist_price"))
        else:
            st.warning("Столбец 'Цена нетто' отсутствует или не является числовым.")
        
        # Статистика по ценам
        st.subheader("Статистика по ценам")
        
        if 'Цена нетто' in data.columns and pd.api.types.is_numeric_dtype(data['Цена нетто']):
            price_stats = data["Цена нетто"].describe()
            
            stats_df = pd.DataFrame({
                "Статистика": price_stats.index,
                "Значение": price_stats.values
            })
            
            st.dataframe(format_streamlit_dataframe(stats_df), use_container_width=True)
            st.markdown(get_general_explanation("table_price_stats"))
        else:
             st.warning("Столбец 'Цена нетто' отсутствует или не является числовым.")
        
        # Анализ пропущенных значений
        st.subheader("Анализ пропущенных значений")
        
        missing_values = data.isna().sum().reset_index()
        missing_values.columns = ["Колонка", "Количество пропущенных значений"]
        missing_values["Процент пропущенных значений"] = (missing_values["Количество пропущенных значений"] / len(data)) * 100
        missing_values = missing_values[missing_values["Количество пропущенных значений"] > 0]
        missing_values = missing_values.sort_values("Количество пропущенных значений", ascending=False)
        
        if not missing_values.empty:
            st.dataframe(format_streamlit_dataframe(missing_values), use_container_width=True)
            st.markdown(get_general_explanation("table_missing_values"))
        else:
            st.success("Пропущенные значения в данных отсутствуют.")
    
    def render_materials_uniqueness(self, data):
        """
        Отображает анализ уникальности материалов
        """
        st.header("Анализ уникальности материалов")
        
        if 'Материал' not in data.columns or 'ДатаСоздан' not in data.columns:
             st.error("Отсутствуют необходимые колонки 'Материал' или 'ДатаСоздан'.")
             return
        
        # Количество записей для каждого материала
        st.subheader("Распределение количества записей по материалам")
        
        # Группируем по материалам и считаем количество записей
        material_counts = data.groupby("Материал")["ДатаСоздан"].count().reset_index()
        material_counts.columns = ["Материал", "Количество записей"]
        material_counts = material_counts.sort_values("Количество записей", ascending=False).reset_index(drop=True)
        
        # Поиск материала по коду
        search_material = st.text_input("Поиск материала по коду или названию:", key="material_search_uniqueness")
        
        # Определяем данные для отображения
        if search_material:
            try:
                 # Используем regex=False для безопасности и производительности, если не нужны регулярные выражения
                 filtered_materials = material_counts[
                     material_counts["Материал"].astype(str).str.contains(search_material, case=False, na=False, regex=False)
                 ]
            except Exception as e:
                 st.error(f"Ошибка при поиске материала: {e}")
                 filtered_materials = pd.DataFrame(columns=material_counts.columns)

            st.write(f"Найдено материалов: {len(filtered_materials)}")
            data_to_show = filtered_materials
        else:
            st.write(f"Топ-20 материалов по количеству записей (всего материалов: "
                     f"{material_counts.shape[0]:,}):".replace(",", " "))
            data_to_show = material_counts.head(20)
        
        # Отображение таблицы
        if not data_to_show.empty:
            st.dataframe(
                format_streamlit_dataframe(data_to_show),
                use_container_width=True,
                height=500
            )
            st.markdown(get_general_explanation("table_material_counts"))
        elif search_material:
            st.info("Материалы, соответствующие поисковому запросу, не найдены.")
        else:
            st.info("Нет данных для отображения.")
        
        # Гистограмма распределения количества записей
        st.subheader("Гистограмма распределения количества записей")
        
        if not material_counts.empty:
            # Ограничиваем для наглядности
            max_records_for_histogram = min(int(material_counts["Количество записей"].max()), 100)
            # Убедимся, что гистограмма имеет хотя бы один бин
            if max_records_for_histogram < 1:
                max_records_for_histogram = 1

            fig = px.histogram(
                material_counts[material_counts["Количество записей"] > 0], # Исключаем материалы без записей, если такие есть
                x="Количество записей",
                nbins=min(50, max_records_for_histogram), # Не больше бинов чем макс значение
                range_x=[1, max_records_for_histogram],
                title=f"Распределение материалов по количеству записей (до {max_records_for_histogram})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("hist_material_records"))
        else:
             st.warning("Нет данных для построения гистограммы.")
        
        # Статистика по количеству записей
        st.subheader("Статистика по количеству записей для материалов")
        
        if not material_counts.empty:
            records_stats = material_counts["Количество записей"].describe()
            
            stats_df = pd.DataFrame({
                "Статистика": records_stats.index,
                "Значение": records_stats.values
            })
            
            st.dataframe(format_streamlit_dataframe(stats_df), use_container_width=True)
            st.markdown(get_general_explanation("table_records_stats"))
            
            # Анализ материалов с одной записью
            single_record_materials = material_counts[material_counts["Количество записей"] == 1]
            
            st.write(f"Материалы с одной записью: {len(single_record_materials)} "
                    f"({len(single_record_materials) / len(material_counts) * 100:.2f}% от всех материалов)")
        else:
             st.warning("Нет данных для расчета статистики.")
        
        # Разделение материалов по количеству записей
        st.subheader("Разделение материалов по количеству записей")
        
        if not material_counts.empty:
            # Создаем категории
            # Убедимся, что максимальное значение больше 0
            if material_counts["Количество записей"].max() > 0:
                bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
                labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", ">100"]

                # Применяем pd.cut
                try:
                    material_counts["Категория"] = pd.cut(
                        material_counts["Количество записей"],
                        bins=bins,
                        labels=labels,
                        right=True,
                        include_lowest=True
                    )
                    # Исправляем NaN, если они возникли (хотя include_lowest должен помочь)
                    material_counts['Категория'] = material_counts['Категория'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                    # Группируем по категориям
                    # Используем observed=True для работы с категориальными данными
                    category_counts = material_counts.groupby("Категория", observed=True)["Материал"].count().reset_index()
                    category_counts.columns = ["Категория", "Количество материалов"]
                    category_counts = category_counts[category_counts['Количество материалов'] > 0] # Убираем пустые категории

                    # Переупорядочиваем категории для графика
                    category_order = [l for l in labels if l in category_counts['Категория'].unique()]
                    if 'Неизвестно' in category_counts['Категория'].unique():
                        category_order.append('Неизвестно')

                    fig = px.bar(
                        category_counts,
                        x="Категория",
                        y="Количество материалов",
                        text="Количество материалов",
                        title="Распределение материалов по категориям количества записей",
                        category_orders={"Категория": category_order} # Задаем порядок
                    )
                    fig.update_traces(texttemplate='%{text:,}', textposition='outside')

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(get_general_explanation("bar_material_categories"))

                except ValueError as e:
                     st.error(f"Ошибка при создании категорий: {e}. Возможно, все материалы имеют 0 записей?")
                except Exception as e:
                     st.error(f"Непредвиденная ошибка при категоризации: {e}")
            else:
                st.info("Нет материалов с количеством записей > 0 для категоризации.")
        else:
             st.warning("Нет данных для разделения материалов.")
    
    def render_time_analysis(self, data):
        """
        Отображает временной анализ данных
        """
        st.header("Временной анализ данных")
        
        if 'Материал' not in data.columns or 'ДатаСоздан' not in data.columns or 'Цена нетто' not in data.columns:
             st.error("Отсутствуют необходимые колонки: 'Материал', 'ДатаСоздан', 'Цена нетто'.")
             return

        unique_materials = data["Материал"].unique()
        if len(unique_materials) == 0:
             st.warning("Нет уникальных материалов для анализа.")
             return

        options = sorted(list(unique_materials))

        selected_material = st.selectbox(
            "Выберите материал для детального анализа:",
            options,
            index=0,
             key="material_select_time"
        )

        if not selected_material:
            st.warning("Пожалуйста, выберите материал для анализа.")
            return

        # Фильтруем данные по выбранному материалу
        material_data = data[data["Материал"] == selected_material].copy()

        if material_data.empty:
             st.info(f"Для материала '{selected_material}' нет данных для анализа.")
             return

        # Проверяем и преобразуем 'ДатаСоздан', если это необходимо
        if not pd.api.types.is_datetime64_any_dtype(material_data['ДатаСоздан']):
            try:
                material_data['ДатаСоздан'] = pd.to_datetime(material_data['ДатаСоздан'], errors='coerce')
                if material_data['ДатаСоздан'].isnull().any():
                     st.warning("Некоторые значения в 'ДатаСоздан' не удалось преобразовать в дату. Эти строки будут проигнорированы.")
                     material_data = material_data.dropna(subset=['ДатаСоздан'])
                     if material_data.empty:
                         st.info("Не осталось данных после удаления некорректных дат.")
                         return
            except Exception as e:
                st.error(f"Не удалось преобразовать столбец 'ДатаСоздан' в дату: {e}")
                return

        # Проверяем 'Цена нетто'
        if not pd.api.types.is_numeric_dtype(material_data['Цена нетто']):
            try:
                material_data['Цена нетто'] = pd.to_numeric(material_data['Цена нетто'], errors='coerce')
                if material_data['Цена нетто'].isnull().any():
                    st.warning("Некоторые значения в 'Цена нетто' не удалось преобразовать в числа. Эти строки будут проигнорированы в ценовом анализе.")
                    # Не удаляем строки, т.к. они могут быть полезны для анализа записей по годам
            except Exception as e:
                 st.error(f"Не удалось преобразовать столбец 'Цена нетто' в число: {e}")
                 # Продолжаем без ценового анализа

        # Сортируем по дате для корректных расчетов и графиков
        material_data = material_data.sort_values("ДатаСоздан").reset_index(drop=True)

        # Добавляем столбцы года и месяца, если их нет
        if 'Год' not in material_data.columns:
            material_data['Год'] = material_data['ДатаСоздан'].dt.year
        if 'Месяц' not in material_data.columns:
            material_data['Месяц'] = material_data['ДатаСоздан'].dt.month

        # Количество записей по годам для выбранного материала
        st.subheader(f"Распределение записей по годам для материала: {selected_material}")
        
        # Группируем по годам
        years_count = material_data.groupby("Год")["ДатаСоздан"].count().reset_index()
        years_count.columns = ["Год", "Количество записей"]
        
        if not years_count.empty:
            fig_years = px.bar(
                years_count,
                x="Год",
                y="Количество записей",
                text="Количество записей",
                title=f"Количество записей по годам"
            )
            fig_years.update_traces(texttemplate='%{text:,}', textposition='outside')
            
            st.plotly_chart(fig_years, use_container_width=True)
            st.markdown(get_material_specific_explanation("bar_years_material", selected_material))
        else:
            st.info("Нет данных о количестве записей по годам для этого материала.")
        
        # График изменения цены во времени
        st.subheader(f"Изменение цены во времени для материала: {selected_material}")
        
        # Используем данные с корректными ценами
        price_data = material_data.dropna(subset=['Цена нетто'])
        if not price_data.empty:
            fig_price_time = px.line(
                price_data,
                x="ДатаСоздан",
                y="Цена нетто",
                title=f"Динамика цены"
            )
            
            # Добавляем точки на график
            fig_price_time.add_trace(
                go.Scatter(
                    x=price_data["ДатаСоздан"],
                    y=price_data["Цена нетто"],
                    mode="markers",
                    marker=dict(size=6),
                    name="Точки данных"
                )
            )
            fig_price_time.update_layout(showlegend=True)
            
            st.plotly_chart(fig_price_time, use_container_width=True)
            st.markdown(get_material_specific_explanation("line_price_time", selected_material))
        else:
            st.info("Нет данных о ценах для построения графика динамики (возможно, все цены некорректны).")
        
        # Сезонность (по месяцам)
        st.subheader(f"Сезонность цен по месяцам для материала: {selected_material}")
        
        if not price_data.empty:
            # Группируем по месяцам
            monthly_prices = price_data.groupby("Месяц")["Цена нетто"].mean().reset_index()
            
            month_names = {
                1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
                7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
            }
            
            monthly_prices["Месяц название"] = monthly_prices["Месяц"].map(month_names)
            monthly_prices = monthly_prices.sort_values("Месяц")
            
            if not monthly_prices.empty:
                fig_season = px.line(
                    monthly_prices,
                    x="Месяц название",
                    y="Цена нетто",
                    markers=True,
                    title=f"Средняя цена по месяцам"
                )
                fig_season.update_layout(xaxis_title="Месяц", yaxis_title="Средняя цена нетто")
                
                st.plotly_chart(fig_season, use_container_width=True)
                st.markdown(get_material_specific_explanation("line_monthly_seasonality", selected_material))
            else:
                 st.info("Недостаточно данных для анализа сезонности по месяцам.")
        else:
             st.info("Нет данных о ценах для анализа сезонности.")
        
        # Тепловая карта по годам и месяцам
        st.subheader(f"Тепловая карта цен по годам и месяцам для материала: {selected_material}")
        
        # Проверяем, достаточно ли данных для построения тепловой карты
        if not price_data.empty and \
           len(price_data["Год"].unique()) > 1 and len(price_data["Месяц"].unique()) > 1:
            
            # Группируем по годам и месяцам
            heatmap_data = price_data.groupby(["Год", "Месяц"])["Цена нетто"].mean().reset_index()
            
            # Создаем сводную таблицу
            heatmap_pivot = heatmap_data.pivot_table(
                values="Цена нетто",
                index="Год",
                columns="Месяц",
                aggfunc="mean"
            )
            
            # Создаем тепловую карту
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[month_names.get(m, m) for m in heatmap_pivot.columns],
                y=heatmap_pivot.index,
                colorscale="Viridis",
                colorbar=dict(title="Ср. Цена"),
                hoverongaps=False
            ))
            
            fig_heatmap.update_layout(
                title=f"Тепловая карта средних цен",
                xaxis_title="Месяц",
                yaxis_title="Год"
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown(get_material_specific_explanation("heatmap_monthly_yearly", selected_material))
        elif not price_data.empty:
            st.info("Недостаточно данных для построения тепловой карты (требуются данные минимум за 2 разных года и 2 разных месяца).")
        else:
             st.info("Нет данных о ценах для построения тепловой карты.")
        
        # Статистика по временным интервалам между записями
        st.subheader(f"Анализ интервалов между записями для материала: {selected_material}")
        
        if len(material_data) > 1:
            # Вычисляем разницу между последовательными датами
            # Убедимся что данные отсортированы по дате
            material_data = material_data.sort_values("ДатаСоздан")
            material_data["Интервал"] = material_data["ДатаСоздан"].diff().dt.days
            
            # Убираем первое значение NaN
            intervals = material_data["Интервал"].dropna()
            
            if not intervals.empty:
                # Статистика по интервалам
                interval_stats = intervals.describe()
                
                stats_df = pd.DataFrame({
                    "Статистика": interval_stats.index,
                    "Значение (дни)": interval_stats.values
                })
                
                st.dataframe(
                    format_streamlit_dataframe(stats_df),
                    use_container_width=True,
                    height=300
                )
                st.markdown(get_material_specific_explanation("table_interval_stats", selected_material))
                
                # Гистограмма интервалов
                fig_interval_hist = px.histogram(
                    intervals,
                    x="Интервал",
                    nbins=30,
                    title=f"Распределение интервалов между записями (дни)"
                )
                fig_interval_hist.update_layout(xaxis_title="Интервал, дни", yaxis_title="Частота")
                
                st.plotly_chart(fig_interval_hist, use_container_width=True)
                st.markdown(get_material_specific_explanation("hist_interval_distribution", selected_material))
            else:
                 st.info("Не удалось рассчитать интервалы между записями.")
        else:
            st.info("Недостаточно данных для анализа временных интервалов (требуется минимум 2 записи).")
    
    def display_basic_stats(self, data):
        """
        Отображает базовую статистику по данным
        """
        st.header("Базовая статистика")
        
        # Получаем базовую статистику по числовым столбцам
        stats_df = data.describe().T
        
        # Добавляем типы данных
        stats_df['Тип данных'] = data.dtypes
        
        # Добавляем количество уникальных значений
        stats_df['Уникальных значений'] = data.nunique()
        
        # Добавляем процент заполненности
        stats_df['Заполненность, %'] = (data.count() / len(data) * 100).round(2)
        
        # Сортируем по заполненности
        stats_df = stats_df.sort_values('Заполненность, %', ascending=False)
        
        # Добавляем все полезные статистики, которые могут помочь в анализе
        stats_df = stats_df.rename(columns={
            'count': 'Количество',
            'mean': 'Среднее',
            'std': 'Стандартное отклонение',
            'min': 'Минимум',
            '25%': '25-й перцентиль',
            '50%': 'Медиана',
            '75%': '75-й перцентиль',
            'max': 'Максимум'
        })
        
        # Отображаем базовую статистику
        st.dataframe(
            format_streamlit_dataframe(stats_df),
            use_container_width=True,
            height=500
        )
        
        return stats_df

    def display_missing_values(self, data):
        """
        Отображает информацию о пропущенных значениях
        """
        st.header("Анализ пропущенных значений")
        
        # Вычисляем суммарную статистику по пропущенным значениям
        missing = pd.DataFrame({
            'Количество пропусков': data.isna().sum(),
            'Процент пропусков': (data.isna().sum() / len(data) * 100).round(2)
        })
        
        # Сортируем по проценту пропусков (по убыванию)
        missing = missing.sort_values('Процент пропусков', ascending=False)
        
        # Отображаем таблицу с пропущенными значениями
        st.dataframe(
            format_streamlit_dataframe(missing),
            use_container_width=True,
            height=400
        )
