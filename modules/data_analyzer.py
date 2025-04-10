import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
            date_range = (data["ДатаСоздан"].max() - data["ДатаСоздан"].min()).days
            st.metric("Временной диапазон", f"{date_range} дней")
        
        # Информация о временном диапазоне
        st.subheader("Временной диапазон")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Начальная дата: {data['ДатаСоздан'].min().strftime('%d.%m.%Y')}")
        
        with col2:
            st.write(f"Конечная дата: {data['ДатаСоздан'].max().strftime('%d.%m.%Y')}")
        
        # Распределение по годам
        st.subheader("Распределение записей по годам")
        
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
        
        # Распределение цен
        st.subheader("Распределение цен")
        
        # Вычисляем квантили для определения границ гистограммы
        q1 = data["Цена нетто"].quantile(0.01)
        q3 = data["Цена нетто"].quantile(0.99)
        
        fig = px.histogram(
            data, 
            x="Цена нетто",
            nbins=50,
            range_x=[q1, q3],  # Убираем выбросы
            title="Распределение цен (без выбросов)"
        )
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Статистика по ценам
        st.subheader("Статистика по ценам")
        
        price_stats = data["Цена нетто"].describe()
        
        stats_df = pd.DataFrame({
            "Статистика": price_stats.index,
            "Значение": price_stats.values
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Анализ пропущенных значений
        st.subheader("Анализ пропущенных значений")
        
        missing_values = data.isna().sum().reset_index()
        missing_values.columns = ["Колонка", "Количество пропущенных значений"]
        missing_values["Процент пропущенных значений"] = (missing_values["Количество пропущенных значений"] / len(data)) * 100
        missing_values = missing_values.sort_values("Количество пропущенных значений", ascending=False)
        
        st.dataframe(missing_values, use_container_width=True)
    
    def render_materials_uniqueness(self, data):
        """
        Отображает анализ уникальности материалов
        """
        st.header("Анализ уникальности материалов")
        
        # Количество записей для каждого материала
        st.subheader("Распределение количества записей по материалам")
        
        # Группируем по материалам и считаем количество записей
        material_counts = data.groupby("Материал")["ДатаСоздан"].count().reset_index()
        material_counts.columns = ["Материал", "Количество записей"]
        material_counts = material_counts.sort_values("Количество записей", ascending=False)
        
        # Добавляем поиск по материалам
        search_material = st.text_input("Поиск материала по коду:")
        
        if search_material:
            filtered_materials = material_counts[material_counts["Материал"].str.contains(search_material)]
            st.dataframe(filtered_materials, use_container_width=True)
        else:
            # Показываем топ материалов
            st.write("Топ-20 материалов по количеству записей:")
            st.dataframe(material_counts.head(20), use_container_width=True)
        
        # Гистограмма распределения количества записей
        st.subheader("Гистограмма распределения количества записей")
        
        # Ограничиваем для наглядности
        max_records_for_histogram = min(material_counts["Количество записей"].max(), 100)
        
        fig = px.histogram(
            material_counts, 
            x="Количество записей",
            nbins=50,
            range_x=[1, max_records_for_histogram],
            title="Распределение материалов по количеству записей"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Статистика по количеству записей
        st.subheader("Статистика по количеству записей для материалов")
        
        records_stats = material_counts["Количество записей"].describe()
        
        stats_df = pd.DataFrame({
            "Статистика": records_stats.index,
            "Значение": records_stats.values
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Анализ материалов с одной записью
        single_record_materials = material_counts[material_counts["Количество записей"] == 1]
        
        st.write(f"Материалы с одной записью: {len(single_record_materials)} "
                f"({len(single_record_materials) / len(material_counts) * 100:.2f}% от всех материалов)")
        
        # Разделение материалов по количеству записей
        st.subheader("Разделение материалов по количеству записей")
        
        # Создаем категории
        bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
        labels = ["1", "2-5", "6-10", "11-20", "21-50", "51-100", ">100"]
        
        material_counts["Категория"] = pd.cut(
            material_counts["Количество записей"], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        category_counts = material_counts.groupby("Категория")["Материал"].count().reset_index()
        category_counts.columns = ["Категория", "Количество материалов"]
        
        fig = px.bar(
            category_counts, 
            x="Категория", 
            y="Количество материалов",
            text="Количество материалов",
            title="Распределение материалов по количеству записей"
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_analysis(self, data):
        """
        Отображает временной анализ данных
        """
        st.header("Временной анализ данных")
        
        # Выбор материала для анализа
        unique_materials = data["Материал"].unique()
        
        if len(unique_materials) > 10000:
            # Если материалов слишком много, показываем только топ по количеству записей
            top_materials = data.groupby("Материал")["ДатаСоздан"].count().sort_values(ascending=False).head(1000).index.tolist()
            selected_material = st.selectbox("Выберите материал для анализа:", top_materials)
        else:
            selected_material = st.selectbox("Выберите материал для анализа:", unique_materials)
        
        # Фильтруем данные по выбранному материалу
        material_data = data[data["Материал"] == selected_material]
        
        # Количество записей по годам для выбранного материала
        st.subheader(f"Распределение записей по годам для материала {selected_material}")
        
        # Группируем по годам
        years_count = material_data.groupby("Год")["ДатаСоздан"].count().reset_index()
        years_count.columns = ["Год", "Количество записей"]
        
        fig = px.bar(
            years_count, 
            x="Год", 
            y="Количество записей",
            text="Количество записей",
            title=f"Количество записей по годам для материала {selected_material}"
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # График изменения цены во времени
        st.subheader(f"Изменение цены во времени для материала {selected_material}")
        
        # Сортируем по дате
        material_data_sorted = material_data.sort_values("ДатаСоздан")
        
        fig = px.line(
            material_data_sorted, 
            x="ДатаСоздан", 
            y="Цена нетто",
            title=f"Изменение цены во времени для материала {selected_material}"
        )
        
        # Добавляем точки на график
        fig.add_trace(
            go.Scatter(
                x=material_data_sorted["ДатаСоздан"], 
                y=material_data_sorted["Цена нетто"],
                mode="markers",
                showlegend=False
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Сезонность (по месяцам)
        st.subheader(f"Сезонность цен по месяцам для материала {selected_material}")
        
        # Группируем по месяцам
        monthly_prices = material_data.groupby("Месяц")["Цена нетто"].mean().reset_index()
        
        month_names = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь",
            7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
        }
        
        monthly_prices["Месяц название"] = monthly_prices["Месяц"].map(month_names)
        monthly_prices = monthly_prices.sort_values("Месяц")
        
        fig = px.line(
            monthly_prices, 
            x="Месяц название", 
            y="Цена нетто",
            markers=True,
            title=f"Средняя цена по месяцам для материала {selected_material}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Тепловая карта по годам и месяцам
        st.subheader(f"Тепловая карта цен по годам и месяцам для материала {selected_material}")
        
        # Проверяем, достаточно ли данных для построения тепловой карты
        if len(material_data["Год"].unique()) > 1 and len(material_data["Месяц"].unique()) > 1:
            # Группируем по годам и месяцам
            heatmap_data = material_data.groupby(["Год", "Месяц"])["Цена нетто"].mean().reset_index()
            
            # Создаем сводную таблицу
            heatmap_pivot = heatmap_data.pivot_table(
                values="Цена нетто", 
                index="Год", 
                columns="Месяц",
                aggfunc="mean"
            )
            
            # Создаем тепловую карту
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[month_names[m] for m in heatmap_pivot.columns],
                y=heatmap_pivot.index,
                colorscale="Viridis",
                colorbar=dict(title="Цена нетто"),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Тепловая карта цен по годам и месяцам для материала {selected_material}",
                xaxis_title="Месяц",
                yaxis_title="Год"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для построения тепловой карты по годам и месяцам")
        
        # Статистика по временным интервалам между записями
        st.subheader(f"Статистика по временным интервалам между записями для материала {selected_material}")
        
        if len(material_data) > 1:
            # Сортируем по дате
            material_data_sorted = material_data.sort_values("ДатаСоздан")
            
            # Вычисляем разницу между последовательными датами
            material_data_sorted["Интервал"] = material_data_sorted["ДатаСоздан"].diff().dt.days
            
            # Статистика по интервалам
            interval_stats = material_data_sorted["Интервал"].describe()
            
            stats_df = pd.DataFrame({
                "Статистика": interval_stats.index,
                "Значение (дни)": interval_stats.values
            })
            
            st.dataframe(stats_df, use_container_width=True)
            
            # Гистограмма интервалов
            fig = px.histogram(
                material_data_sorted, 
                x="Интервал",
                nbins=30,
                title=f"Распределение интервалов между записями для материала {selected_material}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Недостаточно данных для анализа временных интервалов")
