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
    Класс для анализа данных временных рядов
    """
    
    def __init__(self, role_names=None):
        # role_names: dict with keys 'ROLE_ID','ROLE_DATE','ROLE_TARGET','ROLE_QTY','ROLE_CURRENCY','ROLE_RATE'
        self.role_names = role_names or {
            'ROLE_ID': 'ID',
            'ROLE_DATE': 'Дата',
            'ROLE_TARGET': 'Целевая Колонка',
            'ROLE_QTY': 'Количество',
            'ROLE_CURRENCY': 'Валюта',
            'ROLE_RATE': 'Курс'
        }
        # expose convenience attributes
        self.ROLE_ID = self.role_names.get('ROLE_ID')
        self.ROLE_DATE = self.role_names.get('ROLE_DATE')
        self.ROLE_TARGET = self.role_names.get('ROLE_TARGET')
        self.ROLE_QTY = self.role_names.get('ROLE_QTY')
        self.ROLE_CURRENCY = self.role_names.get('ROLE_CURRENCY')
        self.ROLE_RATE = self.role_names.get('ROLE_RATE')

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
            unique_materials = data[self.ROLE_ID].nunique() if self.ROLE_ID in data.columns else 0
            st.metric("Уникальных временных рядов", f"{unique_materials:,}".replace(",", " "))
        
        with col3:
            if pd.api.types.is_datetime64_any_dtype(data[self.ROLE_DATE]):
                date_range = (data[self.ROLE_DATE].max() - data[self.ROLE_DATE].min()).days
                st.metric("Временной диапазон", f"{date_range} дней")
            else:
                st.metric("Временной диапазон", "N/A")

        # Информация о временном диапазоне
        st.subheader("Временной диапазон")
        
        col1, col2 = st.columns(2)
        if pd.api.types.is_datetime64_any_dtype(data[self.ROLE_DATE]):
            with col1:
                st.write(f"Начальная дата: {data[self.ROLE_DATE].min().strftime('%d.%m.%Y')}")
            with col2:
                st.write(f"Конечная дата: {data[self.ROLE_DATE].max().strftime('%d.%m.%Y')}")
        else:
             st.warning(f"Столбец '{self.ROLE_DATE}' не является датой.")

        # Распределение по годам
        st.subheader("Распределение данных по годам")

        if 'Год' in data.columns:
            years_count = data.groupby("Год")[self.ROLE_ID].count().reset_index()
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
        if self.ROLE_CURRENCY in data.columns:
            st.subheader("Распределение по валютам")
            
            # Улучшенная обработка валют для графика
            total_records = len(data)
            currency_counts = data[self.ROLE_CURRENCY].value_counts()
            currency_perc = (currency_counts / total_records) * 100

            # Определяем порог для группировки (например, 0.5%)
            threshold = 0.5
            small_currencies = currency_perc[currency_perc < threshold]
            main_currencies = currency_perc[currency_perc >= threshold]

            # Создаем данные для графика
            plot_data_list = []
            # Добавляем основные валюты
            for currency, count in currency_counts[main_currencies.index].items():
                plot_data_list.append({"Валюта": currency, "Количество записей": count})
                
            # Добавляем 'Прочее', если есть мелкие валюты
            if not small_currencies.empty:
                other_count = currency_counts[small_currencies.index].sum()
                plot_data_list.append({"Валюта": "Прочее", "Количество записей": other_count})
                
            plot_df = pd.DataFrame(plot_data_list)
            plot_df = plot_df.sort_values("Количество записей", ascending=False)

            fig = px.pie(
                plot_df, 
                names="Валюта", 
                values="Количество записей",
                title="Распределение данных по валютам (мелкие < {threshold}% сгруппированы)".format(threshold=threshold),
                hole=0.3 # Делаем "бублик" для лучшей читаемости
            )
            # Улучшаем отображение текста
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                insidetextorientation='radial' # Попробуем радиальное расположение
            )
            fig.update_layout(
                height=450, # Немного увеличим высоту
                showlegend=True, # Показываем легенду
                legend=dict(
                    orientation="h", # Горизонтальная легенда
                    yanchor="bottom",
                    y=-0.2, # Размещаем под графиком
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("pie_currency"))
        else:
            st.info(f"Столбец '{self.ROLE_CURRENCY}' (Валюта) отсутствует в данных.")

        # Распределение значений целевой колонки
        st.subheader("Распределение значений целевой колонки")

        norm_col = f"{self.ROLE_TARGET} (норм.)"
        if norm_col in data.columns and pd.api.types.is_numeric_dtype(data[norm_col]):
            # Вычисляем квантили для определения границ гистограммы
            q_low = data[norm_col].quantile(0.01)
            q_high = data[norm_col].quantile(0.99)
            # Убедимся, что границы разумны
            if q_low >= q_high:
                 q_low = data[norm_col].min()
                 q_high = data[norm_col].max()

            fig = px.histogram(
                data[data[norm_col].between(q_low, q_high)], # Фильтруем выбросы здесь
                x=norm_col,
                nbins=50,
                # range_x=[q_low, q_high], # Убираем range_x, так как данные уже отфильтрованы
                title="Распределение значений целевой колонки (1% и 99% перцентили)"
            )
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("hist_price"))
        else:
            st.warning(f"Столбец '{norm_col}' отсутствует или не является числовым.")

        # Статистика по значениям целевой колонки
        st.subheader("Статистика по значениям целевой колонки")

        if norm_col in data.columns and pd.api.types.is_numeric_dtype(data[norm_col]):
            price_stats = data[norm_col].describe()

            stats_df = pd.DataFrame({
                "Статистика": price_stats.index,
                "Значение": price_stats.values
            })

            st.dataframe(format_streamlit_dataframe(stats_df), use_container_width=True)
            st.markdown(get_general_explanation("table_price_stats"))
        else:
             st.warning(f"Столбец '{norm_col}' отсутствует или не является числовым.")

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
        Отображает анализ уникальности временных рядов
        """
        st.header("Анализ уникальности временных рядов")
        
        if self.ROLE_ID not in data.columns or self.ROLE_DATE not in data.columns:
             st.error(f"Отсутствуют необходимые колонки '{self.ROLE_ID}' или '{self.ROLE_DATE}'.")
             return
        
        # Количество записей для каждого временного ряда
        st.subheader("Распределение количества записей по временным рядам")
        
        # Группируем по временным рядам и считаем количество записей
        material_counts = data.groupby(self.ROLE_ID)[self.ROLE_DATE].count().reset_index()
        material_counts.columns = [self.ROLE_ID, "Количество записей"]
        material_counts = material_counts.sort_values("Количество записей", ascending=False).reset_index(drop=True)
        
        # Показываем топ-20 временных рядов по количеству записей
        st.write(f"Топ-20 временных рядов по количеству записей (всего временных рядов: {material_counts.shape[0]:,}):".replace(",", " "))
        st.dataframe(
            format_streamlit_dataframe(material_counts.head(20)),
            use_container_width=True,
            height=500
        )
        # Пояснение для таблицы (без конкретного выбранного ряда)
        st.markdown(get_material_specific_explanation("table_material_counts", None))
        
        # Гистограмма распределения количества записей
        st.subheader("Гистограмма распределения количества записей")

        if not material_counts.empty:
            # Ограничиваем для наглядности
            max_records_for_histogram = min(int(material_counts["Количество записей"].max()), 100)
            # Убедимся, что гистограмма имеет хотя бы один бин
            if max_records_for_histogram < 1:
                max_records_for_histogram = 1

            fig = px.histogram(
                material_counts[material_counts["Количество записей"] > 0], # Исключаем ряды без записей, если такие есть
                x="Количество записей",
                nbins=min(50, max_records_for_histogram), # Не больше бинов чем макс значение
                range_x=[1, max_records_for_histogram],
                title=f"Распределение временных рядов по категории количества записей (до {max_records_for_histogram})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(get_general_explanation("hist_material_records"))
        else:
             st.warning("Нет данных для построения гистограммы.")
        
        # Статистика по количеству записей
        st.subheader("Статистика по количеству записей для временных рядов")

        if not material_counts.empty:
            records_stats = material_counts["Количество записей"].describe()

            stats_df = pd.DataFrame({
                "Статистика": records_stats.index,
                "Значение": records_stats.values
            })

            st.dataframe(format_streamlit_dataframe(stats_df), use_container_width=True)
            st.markdown(get_general_explanation("table_records_stats"))
            
            # Анализ временных рядов с одной записью
            single_record_materials = material_counts[material_counts["Количество записей"] == 1]
            
            st.write(f"Временные ряды с одной записью: {len(single_record_materials)} "
                     f"({len(single_record_materials) / len(material_counts) * 100:.2f}% от всех временных рядов)")
        else:
             st.warning("Нет данных для расчета статистики.")
        
        # Разделение временных рядов по количеству записей
        st.subheader("Разделение временных рядов по количеству записей")

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
                    category_counts = material_counts.groupby("Категория", observed=True)[self.ROLE_ID].count().reset_index()
                    category_counts.columns = ["Категория", "Количество временных рядов"]
                    category_counts = category_counts[category_counts['Количество временных рядов'] > 0] # Убираем пустые категории

                    # Переупорядочиваем категории для графика
                    category_order = [l for l in labels if l in category_counts['Категория'].unique()]
                    if 'Неизвестно' in category_counts['Категория'].unique():
                        category_order.append('Неизвестно')

                    fig = px.bar(
                        category_counts,
                        x="Категория",
                        y="Количество временных рядов",
                        text="Количество временных рядов",
                        title="Распределение временных рядов по категориям количества записей",
                        category_orders={"Категория": category_order} # Задаем порядок
                    )
                    fig.update_traces(texttemplate='%{text:,}', textposition='outside')

                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(get_general_explanation("bar_material_categories"))

                except ValueError as e:
                     st.error(f"Ошибка при создании категорий: {e}. Возможно, все временные ряды имеют 0 записей?")
                except Exception as e:
                     st.error(f"Непредвиденная ошибка при категоризацией: {e}")
            else:
                st.info("Нет временных рядов с количеством записей > 0 для категоризации.")
        else:
             st.warning("Нет данных для разделения временных рядов.")

    def render_time_analysis(self, data):
        """
        Отображает временной анализ данных
        """
        st.header("Временной анализ данных")
        
        norm_col = f"{self.ROLE_TARGET} (норм.)"
        if self.ROLE_ID not in data.columns or self.ROLE_DATE not in data.columns or norm_col not in data.columns:
             st.error(f"Отсутствуют необходимые колонки: '{self.ROLE_ID}', '{self.ROLE_DATE}', '{norm_col}'.")
             return

        unique_materials = data[self.ROLE_ID].unique()
        if len(unique_materials) == 0:
             st.warning("Нет уникальных временных рядов для анализа.")
             return

        options = sorted(list(unique_materials))

        selected_material = st.selectbox(
            "Выберите временной ряд для детального анализа:",
            options,
            index=0,
             key="material_select_time"
        )

        if not selected_material:
            st.warning("Пожалуйста, выберите временной ряд для анализа.")
            return

        # Фильтруем данные по выбранному временному ряду
        material_data = data[data[self.ROLE_ID] == selected_material].copy()

        if material_data.empty:
             st.info(f"Для временного ряда '{selected_material}' нет данных для анализа.")
             return

        # Проверяем и преобразуем 'Дата' , если это необходимо
        if not pd.api.types.is_datetime64_any_dtype(material_data[self.ROLE_DATE]):
            try:
                material_data[self.ROLE_DATE] = pd.to_datetime(material_data[self.ROLE_DATE], errors='coerce')
                if material_data[self.ROLE_DATE].isnull().any():
                     st.warning(f"Некоторые значения в '{self.ROLE_DATE}' не удалось преобразовать в дату. Эти строки будут проигнорированы.")
                     material_data = material_data.dropna(subset=[self.ROLE_DATE])
                     if material_data.empty:
                         st.info("Не осталось данных после удаления некорректных дат.")
                         return
            except Exception as e:
                st.error(f"Не удалось преобразовать столбец '{self.ROLE_DATE}' в дату: {e}")
                return

        # Проверяем нормализованные значения
        if not pd.api.types.is_numeric_dtype(material_data[norm_col]):
            try:
                material_data[norm_col] = pd.to_numeric(material_data[norm_col], errors='coerce')
                if material_data[norm_col].isnull().any():
                    st.warning(f"Некоторые значения в '{norm_col}' не удалось преобразовать в числа. Эти строки будут проигнорированы в анализе значений.")
                    # Не удаляем строки, т.к. они могут быть полезны для анализа записей по годам
            except Exception as e:
                 st.error(f"Не удалось преобразовать столбец '{norm_col}' в число: {e}")
                 # Продолжаем без анализа значений

        # Сортируем по дате для корректных расчетов и графиков
        material_data = material_data.sort_values(self.ROLE_DATE).reset_index(drop=True)

        # Добавляем столбцы года и месяца, если их нет
        if 'Год' not in material_data.columns:
            material_data['Год'] = material_data[self.ROLE_DATE].dt.year
        if 'Месяц' not in material_data.columns:
            material_data['Месяц'] = material_data[self.ROLE_DATE].dt.month

        # Количество записей по годам для выбранного временного ряда
        st.subheader(f"Распределение данных по годам для временного ряда: {selected_material}")
        
        # Группируем по годам
        years_count = material_data.groupby("Год")[self.ROLE_DATE].count().reset_index()
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
            st.info("Нет данных о количестве записей по годам для этого временного ряда.")
        
        # График изменения целевого значения во времени
        st.subheader(f"Изменение целевого значения во времени для временного ряда: {selected_material}")
        
        # Используем данные с корректными нормализованными значениями
        price_data = material_data.dropna(subset=[norm_col]).copy()
        if not price_data.empty:
            fig_price_time = px.line(
                price_data,
                x=self.ROLE_DATE,
                y=norm_col,
                title=f"Динамика целевого значения"
            )

            # Добавляем точки на график
            fig_price_time.add_trace(
                go.Scatter(
                    x=price_data[self.ROLE_DATE],
                    y=price_data[norm_col],
                    mode="markers",
                    marker=dict(size=6),
                    name="Точки данных"
                )
            )
            fig_price_time.update_layout(showlegend=True)

            st.plotly_chart(fig_price_time, use_container_width=True)
            st.markdown(get_material_specific_explanation("line_price_time", selected_material))
        else:
            st.info("Нет данных о целевых значениях для построения графика динамики (возможно, все значения некорректны).")
        
        # Сезонность (по месяцам)
        st.subheader(f"Сезонность по месяцам для временного ряда: {selected_material}")
        
        if not price_data.empty:
            # Группируем по месяцам
            monthly_prices = price_data.groupby("Месяц")[norm_col].mean().reset_index()

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
                    y=norm_col,
                    markers=True,
                    title=f"Среднее целевое значение по месяцам"
                )
                fig_season.update_layout(xaxis_title="Месяц", yaxis_title=f"Среднее значение")

                st.plotly_chart(fig_season, use_container_width=True)
                st.markdown(get_material_specific_explanation("line_monthly_seasonality", selected_material))
            else:
                st.info("Недостаточно данных для анализа сезонности по месяцам.")
        else:
             st.info("Нет данных о целевых значениях для анализа сезонности.")
        
        # Тепловая карта по годам и месяцам
        st.subheader(f"Тепловая карта целевых значений по годам и месяцам для временного ряда: {selected_material}")

        # Проверяем, достаточно ли данных для построения тепловой карты
        if not price_data.empty and len(price_data["Год"].unique()) > 1 and len(price_data["Месяц"].unique()) > 1:
            # Группируем по годам и месяцам
            heatmap_data = price_data.groupby(["Год", "Месяц"])[norm_col].mean().reset_index()

            # Создаем сводную таблицу
            heatmap_pivot = heatmap_data.pivot_table(
                values=norm_col,
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
                colorbar=dict(title="Ср. значение"),
                hoverongaps=False
            ))

            fig_heatmap.update_layout(
                title=f"Тепловая карта средних значений",
                xaxis_title="Месяц",
                yaxis_title="Год"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.markdown(get_material_specific_explanation("heatmap_monthly_yearly", selected_material))
        elif not price_data.empty:
            st.info("Недостаточно данных для построения тепловой карты (требуются данные минимум за 2 разных года и 2 разных месяца).")
        else:
            st.info("Нет данных о целевых значениях для построения тепловой карты.")
        
        # Статистика по временным интервалам между записями
        st.subheader(f"Анализ интервалов между записями для временного ряда: {selected_material}")
        
        if len(material_data) > 1:
            # Вычисляем разницу между последовательными датами
            # Убедимся что данные отсортированы по дате
            material_data = material_data.sort_values(self.ROLE_DATE)
            material_data["Интервал"] = material_data[self.ROLE_DATE].diff().dt.days
            
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