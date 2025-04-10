import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.utils import create_styled_dataframe

class Visualizer:
    """
    Класс для визуализации данных о материалах
    """
    
    def __init__(self):
        pass
    
    def plot_materials_distribution(self, data):
        """
        Визуализирует распределение материалов
        """
        st.subheader("Визуализация распределения материалов")
        
        # Распределение материалов по количеству записей
        materials_count = data.groupby('Материал')['ДатаСоздан'].count().reset_index()
        materials_count.columns = ['Материал', 'Количество записей']
        materials_count = materials_count.sort_values('Количество записей', ascending=False)
        
        # Отображаем топ-20 материалов
        top_materials = materials_count.head(20)
        
        fig = px.bar(
            top_materials,
            x='Материал',
            y='Количество записей',
            text='Количество записей',
            title='Топ-20 материалов по количеству записей'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,  # Устанавливаем фиксированную высоту
            margin=dict(l=20, r=20, t=40, b=20),  # Уменьшаем отступы
            autosize=True  # Разрешаем автоматическое изменение размера
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Распределение материалов по группам
        if 'ГруппаМтр' in data.columns:
            group_counts = data.groupby('ГруппаМтр')['Материал'].nunique().reset_index()
            group_counts.columns = ['Группа материала', 'Количество уникальных материалов']
            group_counts = group_counts.sort_values('Количество уникальных материалов', ascending=False)
            
            # Отображаем топ-10 групп
            top_groups = group_counts.head(10)
            
            fig = px.bar(
                top_groups,
                x='Группа материала',
                y='Количество уникальных материалов',
                text='Количество уникальных материалов',
                title='Топ-10 групп по количеству уникальных материалов'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Круговая диаграмма распределения материалов по количеству записей
        st.subheader("Распределение материалов по количеству записей")
        
        # Создаем категории
        bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
        labels = ["1 запись", "2-5 записей", "6-10 записей", "11-20 записей", 
                 "21-50 записей", "51-100 записей", ">100 записей"]
        
        materials_count['Категория'] = pd.cut(
            materials_count['Количество записей'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        category_counts = materials_count.groupby('Категория')['Материал'].count().reset_index()
        category_counts.columns = ['Категория', 'Количество материалов']
        
        fig = px.pie(
            category_counts,
            names='Категория',
            values='Количество материалов',
            title='Распределение материалов по количеству записей'
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_time_distribution(self, data):
        """
        Визуализирует временное распределение данных
        """
        st.subheader("Визуализация временного распределения")
        
        # Временной ряд общего количества записей по месяцам
        monthly_data = data.groupby(pd.Grouper(key='ДатаСоздан', freq='M')).size().reset_index()
        monthly_data.columns = ['Дата', 'Количество записей']
        
        fig = px.line(
            monthly_data,
            x='Дата',
            y='Количество записей',
            title='Количество записей по месяцам'
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Тепловая карта по годам и месяцам
        st.subheader("Тепловая карта количества записей по годам и месяцам")
        
        # Добавляем год и месяц как отдельные колонки
        data_heatmap = data.copy()
        data_heatmap['Год'] = data_heatmap['ДатаСоздан'].dt.year
        data_heatmap['Месяц'] = data_heatmap['ДатаСоздан'].dt.month
        
        # Группируем по годам и месяцам
        heatmap_data = data_heatmap.groupby(['Год', 'Месяц']).size().reset_index()
        heatmap_data.columns = ['Год', 'Месяц', 'Количество записей']
        
        # Создаем сводную таблицу
        heatmap_pivot = heatmap_data.pivot_table(
            values='Количество записей', 
            index='Год', 
            columns='Месяц',
            aggfunc='sum'
        )
        
        # Названия месяцев
        month_names = {
            1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
            7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
        }
        
        # Создаем тепловую карту
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[month_names[m] for m in heatmap_pivot.columns],
            y=heatmap_pivot.index,
            colorscale="Viridis",
            colorbar=dict(title="Кол-во записей"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Тепловая карта количества записей по годам и месяцам",
            xaxis_title="Месяц",
            yaxis_title="Год",
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Сезонный график средних цен по месяцам
        st.subheader("Сезонность средних цен по месяцам")
        
        monthly_prices = data.groupby(['Год', 'Месяц'])['Цена нетто'].mean().reset_index()
        
        # Добавляем названия месяцев
        monthly_prices['Месяц название'] = monthly_prices['Месяц'].map(month_names)
        
        # Строим график для каждого года
        fig = px.line(
            monthly_prices,
            x='Месяц',
            y='Цена нетто',
            color='Год',
            markers=True,
            title='Сезонность средних цен по месяцам',
            labels={'Месяц': 'Месяц', 'Цена нетто': 'Средняя цена нетто'}
        )
        
        # Настраиваем оси
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=[month_names[m] for m in range(1, 13)]
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_volatility(self, volatility_data):
        """
        Визуализирует волатильность цен материалов
        """
        st.subheader("Визуализация волатильности цен материалов")
        
        # Топ-20 материалов с наибольшей волатильностью
        top_volatile = volatility_data.head(20)
        
        fig = px.bar(
            top_volatile,
            x='Материал',
            y='Коэффициент вариации',
            text='Коэффициент вариации',
            title='Топ-20 материалов с наибольшей волатильностью цен'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Топ-20 материалов с наименьшей волатильностью (но не нулевой)
        bottom_volatile = volatility_data[volatility_data['Коэффициент вариации'] > 0].tail(20)
        
        fig = px.bar(
            bottom_volatile,
            x='Материал',
            y='Коэффициент вариации',
            text='Коэффициент вариации',
            title='Топ-20 материалов с наименьшей ненулевой волатильностью цен'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Распределение коэффициентов вариации
        st.subheader("Распределение коэффициентов вариации")
        
        # Ограничиваем до 100% для наглядности
        histogram_data = volatility_data[volatility_data['Коэффициент вариации'] <= 100]
        
        fig = px.histogram(
            histogram_data,
            x='Коэффициент вариации',
            nbins=50,
            title='Распределение коэффициентов вариации цен материалов'
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Категоризация материалов по волатильности
        st.subheader("Категоризация материалов по волатильности")
        
        # Создаем категории
        bins = [0, 5, 10, 20, 30, 50, 100, np.inf]
        labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]
        
        volatility_data['Категория волатильности'] = pd.cut(
            volatility_data['Коэффициент вариации'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        category_counts = volatility_data.groupby('Категория волатильности')['Материал'].count().reset_index()
        category_counts.columns = ['Категория волатильности', 'Количество материалов']
        
        fig = px.pie(
            category_counts,
            names='Категория волатильности',
            values='Количество материалов',
            title='Распределение материалов по категориям волатильности'
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_stability(self, stability_data):
        """
        Визуализирует стабильность цен материалов
        """
        st.subheader("Визуализация стабильных цен материалов")
        
        # Количество материалов со стабильными и нестабильными ценами
        stability_counts = stability_data['Стабильная цена'].value_counts().reset_index()
        stability_counts.columns = ['Стабильная цена', 'Количество материалов']
        
        # Заменяем булевы значения на текст
        stability_counts['Стабильная цена'] = stability_counts['Стабильная цена'].map({
            True: 'Стабильная цена (≥80% одинаковых значений)',
            False: 'Нестабильная цена (<80% одинаковых значений)'
        })
        
        fig = px.pie(
            stability_counts,
            names='Стабильная цена',
            values='Количество материалов',
            title='Распределение материалов по стабильности цен'
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Топ-20 материалов с наибольшим процентом одинаковых значений
        st.subheader("Топ материалов с наибольшим процентом одинаковых значений")
        
        # Фильтруем только материалы с несколькими записями
        filtered_data = stability_data[stability_data['Количество записей'] > 1].head(20)
        
        fig = px.bar(
            filtered_data,
            x='Материал',
            y='Процент одинаковых значений',
            text='Процент одинаковых значений',
            title='Топ-20 материалов с наибольшим процентом одинаковых значений цены'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Соотношение между количеством записей и стабильностью цен
        st.subheader("Соотношение между количеством записей и стабильностью цен")
        
        # Ограничиваем количество записей для наглядности
        scatter_data = stability_data[stability_data['Количество записей'] <= 100].copy()
        
        # Добавляем цветовую маркировку для стабильных и нестабильных цен
        scatter_data['Статус'] = scatter_data['Стабильная цена'].map({
            True: 'Стабильная цена',
            False: 'Нестабильная цена'
        })
        
        fig = px.scatter(
            scatter_data,
            x='Количество записей',
            y='Процент одинаковых значений',
            color='Статус',
            hover_name='Материал',
            title='Соотношение между количеством записей и стабильностью цен',
            labels={'Процент одинаковых значений': 'Процент одинаковых значений цены (%)'}
        )
        
        # Добавляем горизонтальную линию на уровне 80%
        fig.add_shape(
            type="line",
            x0=0,
            y0=80,
            x1=100,
            y1=80,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_inactivity(self, inactivity_data):
        """
        Визуализирует неактивные материалы
        """
        st.subheader("Визуализация неактивных материалов")
        
        # Создаем категории по времени неактивности
        bins = [0, 30, 90, 180, 365, 730, np.inf]
        labels = ["<30 дней", "30-90 дней", "90-180 дней", "180-365 дней", "1-2 года", ">2 года"]
        
        inactivity_data['Категория неактивности'] = pd.cut(
            inactivity_data['Дней с последней активности'], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        
        category_counts = inactivity_data.groupby('Категория неактивности')['Материал'].count().reset_index()
        category_counts.columns = ['Категория неактивности', 'Количество материалов']
        
        fig = px.bar(
            category_counts,
            x='Категория неактивности',
            y='Количество материалов',
            text='Количество материалов',
            title='Распределение материалов по времени неактивности'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Круговая диаграмма активных и неактивных материалов
        st.subheader("Активные и неактивные материалы")
        
        activity_counts = inactivity_data['Неактивный материал'].value_counts().reset_index()
        activity_counts.columns = ['Статус', 'Количество материалов']
        
        # Заменяем булевы значения на текст
        activity_counts['Статус'] = activity_counts['Статус'].map({
            True: 'Неактивный материал (>365 дней)',
            False: 'Активный материал (≤365 дней)'
        })
        
        fig = px.pie(
            activity_counts,
            names='Статус',
            values='Количество материалов',
            title='Распределение материалов по активности'
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Топ-20 материалов с наибольшим периодом неактивности
        st.subheader("Топ материалов с наибольшим периодом неактивности")
        
        top_inactive = inactivity_data.sort_values('Дней с последней активности', ascending=False).head(20)
        
        fig = px.bar(
            top_inactive,
            x='Материал',
            y='Дней с последней активности',
            text='Дней с последней активности',
            title='Топ-20 материалов с наибольшим периодом неактивности'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # График распределения последних дат активности
        st.subheader("Распределение последних дат активности")
        
        # Группируем по месяцам последней активности
        last_activity = inactivity_data.groupby(pd.Grouper(key='Последняя активность материала', freq='M')).size().reset_index()
        last_activity.columns = ['Дата', 'Количество материалов']
        
        fig = px.bar(
            last_activity,
            x='Дата',
            y='Количество материалов',
            title='Распределение материалов по дате последней активности'
        )
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_segmentation_results(self, segments, stats):
        """
        Визуализирует результаты сегментации материалов
        """
        st.subheader("Результаты сегментации материалов")
        
        # Общие результаты сегментации
        segment_counts = pd.DataFrame({
            'Сегмент': list(stats.keys()),
            'Количество материалов': list(stats.values())
        })
        
        fig = px.bar(
            segment_counts,
            x='Сегмент',
            y='Количество материалов',
            text='Количество материалов',
            title='Распределение материалов по сегментам'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Круговая диаграмма распределения по сегментам
        fig = px.pie(
            segment_counts,
            names='Сегмент',
            values='Количество материалов',
            title='Распределение материалов по сегментам'
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Детализация по сегментам
        st.subheader("Детализация по сегментам")
        
        # Объединяем все сегменты в один DataFrame
        all_segments = pd.DataFrame()
        
        for segment, data in segments.items():
            if not data.empty:
                data_copy = data.copy()
                data_copy['Сегмент'] = segment
                all_segments = pd.concat([all_segments, data_copy])
        
        # Выбор сегмента для просмотра
        segment_selection = st.selectbox(
            "Выберите сегмент для просмотра:",
            ['Все сегменты'] + list(segments.keys())
        )
        
        if segment_selection == 'Все сегменты':
            display_data = all_segments
        else:
            display_data = segments[segment_selection]
        
        # Показываем таблицу с данными выбранного сегмента
        st.write(f"Данные сегмента: {segment_selection}")
        st.dataframe(
            create_styled_dataframe(
                display_data,
                precision=2
            ),
            use_container_width=True
        )
        
        # Визуализация характеристик сегментов
        if not all_segments.empty:
            st.subheader("Характеристики сегментов")
            
            # Проверяем наличие необходимых колонок
            if 'Количество записей материала' in all_segments.columns:
                # График среднего количества записей по сегментам
                records_by_segment = all_segments.groupby('Сегмент')['Количество записей материала'].mean().reset_index()
                records_by_segment.columns = ['Сегмент', 'Среднее количество записей']
                
                fig = px.bar(
                    records_by_segment,
                    x='Сегмент',
                    y='Среднее количество записей',
                    text='Среднее количество записей',
                    title='Среднее количество записей по сегментам'
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    autosize=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # График средней волатильности по сегментам
            if 'Коэффициент вариации цены' in all_segments.columns:
                volatility_by_segment = all_segments.groupby('Сегмент')['Коэффициент вариации цены'].mean().reset_index()
                volatility_by_segment.columns = ['Сегмент', 'Средний коэффициент вариации']
                
                fig = px.bar(
                    volatility_by_segment,
                    x='Сегмент',
                    y='Средний коэффициент вариации',
                    text='Средний коэффициент вариации',
                    title='Средний коэффициент вариации по сегментам'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    autosize=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Диаграмма распределения временных диапазонов по сегментам
            if 'Временной диапазон материала' in all_segments.columns:
                timerange_by_segment = all_segments.groupby('Сегмент')['Временной диапазон материала'].mean().reset_index()
                timerange_by_segment.columns = ['Сегмент', 'Средний временной диапазон (дни)']
                
                fig = px.bar(
                    timerange_by_segment,
                    x='Сегмент',
                    y='Средний временной диапазон (дни)',
                    text='Средний временной диапазон (дни)',
                    title='Средний временной диапазон материалов по сегментам'
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    autosize=True
                )
                
                st.plotly_chart(fig, use_container_width=True)