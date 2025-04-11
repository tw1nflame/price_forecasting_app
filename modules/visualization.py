import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.utils import create_styled_dataframe, get_general_explanation

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
        if 'Материал' not in data.columns or 'ДатаСоздан' not in data.columns:
             st.error("Отсутствуют необходимые колонки 'Материал' или 'ДатаСоздан'.")
             return

        # Распределение материалов по количеству записей
        materials_count = data.groupby('Материал')['ДатаСоздан'].count().reset_index()
        materials_count.columns = ['Материал', 'Количество записей']
        materials_count = materials_count.sort_values('Количество записей', ascending=False)
        
        # Отображаем топ-20 материалов
        if not materials_count.empty:
             top_materials = materials_count.head(20)

             fig_top_mat = px.bar(
                 top_materials,
                 x='Материал',
                 y='Количество записей',
                 text='Количество записей',
                 title='Топ-20 материалов по количеству записей'
             )
             fig_top_mat.update_traces(texttemplate='%{text:,}', textposition='outside')
             fig_top_mat.update_layout(
                 xaxis_tickangle=-45,
                 height=500,
                 margin=dict(l=20, r=20, t=40, b=20),
                 autosize=True
             )

             st.plotly_chart(fig_top_mat, use_container_width=True)
             st.markdown(get_general_explanation("bar_top_materials_records"))
        else:
             st.info("Нет данных для отображения топ материалов.")
        
        # Распределение материалов по группам
        if 'ГруппаМтр' in data.columns:
            group_counts = data.groupby('ГруппаМтр')['Материал'].nunique().reset_index()
            group_counts.columns = ['Группа материала', 'Количество уникальных материалов']
            group_counts = group_counts.sort_values('Количество уникальных материалов', ascending=False)
            
            if not group_counts.empty:
                 # Отображаем топ-10 групп
                 top_groups = group_counts.head(10)

                 fig_top_grp = px.bar(
                     top_groups,
                     x='Группа материала',
                     y='Количество уникальных материалов',
                     text='Количество уникальных материалов',
                     title='Топ-10 групп по количеству уникальных материалов'
                 )
                 fig_top_grp.update_traces(texttemplate='%{text:,}', textposition='outside')
                 fig_top_grp.update_layout(
                     xaxis_tickangle=-45,
                     height=500,
                     margin=dict(l=20, r=20, t=40, b=20),
                     autosize=True
                 )

                 st.plotly_chart(fig_top_grp, use_container_width=True)
                 st.markdown(get_general_explanation("bar_top_material_groups"))
            else:
                 st.info("Нет данных о группах материалов.")
        else:
             st.info("Столбец 'ГруппаМтр' отсутствует, распределение по группам недоступно.")
        
        # Круговая диаграмма распределения материалов по количеству записей
        st.subheader("Распределение материалов по количеству записей")
        
        if not materials_count.empty:
            # Создаем категории
            if materials_count['Количество записей'].max() > 0:
                bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
                labels = ["1", "2-5", "6-10", "11-20",
                         "21-50", "51-100", ">100"]

                try:
                    materials_count['Категория'] = pd.cut(
                        materials_count['Количество записей'],
                        bins=bins,
                        labels=labels,
                        right=True,
                        include_lowest=True
                    )
                    materials_count['Категория'] = materials_count['Категория'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                    category_counts = materials_count.groupby('Категория', observed=True)['Материал'].count().reset_index()
                    category_counts.columns = ['Категория', 'Количество материалов']
                    category_counts = category_counts[category_counts['Количество материалов'] > 0]

                    if not category_counts.empty:
                        fig_cat_pie = px.pie(
                            category_counts,
                            names='Категория',
                            values='Количество материалов',
                            title='Доля материалов по категориям количества записей'
                        )
                        fig_cat_pie.update_traces(textinfo='percent+label')
                        fig_cat_pie.update_layout(
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20),
                            autosize=True
                        )

                        st.plotly_chart(fig_cat_pie, use_container_width=True)
                        st.markdown(get_general_explanation("pie_material_records_categories"))
                    else:
                         st.info("Нет данных для построения круговой диаграммы категорий.")

                except Exception as e:
                     st.error(f"Ошибка при создании категорий для диаграммы: {e}")
            else:
                st.info("Нет материалов с >0 записей для категоризации.")
        else:
             st.info("Нет данных о количестве записей материалов.")
    
    def plot_time_distribution(self, data):
        """
        Визуализирует временное распределение данных
        """
        st.subheader("Визуализация временного распределения")
        if 'ДатаСоздан' not in data.columns or not pd.api.types.is_datetime64_any_dtype(data['ДатаСоздан']):
             st.error("Отсутствует или имеет неверный формат колонка 'ДатаСоздан'.")
             return

        # Временной ряд общего количества записей по месяцам
        data_ts = data.set_index('ДатаСоздан')
        # Указываем частоту 'ME' для группировки по концу месяца
        monthly_data = data_ts.resample('ME').size().reset_index()
        monthly_data.columns = ['Дата', 'Количество записей']
        
        if not monthly_data.empty:
            fig_ts_records = px.line(
                monthly_data,
                x='Дата',
                y='Количество записей',
                title='Общее количество записей по месяцам'
            )
            fig_ts_records.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_ts_records, use_container_width=True)
            st.markdown(get_general_explanation("line_monthly_records"))
        else:
             st.info("Нет данных для построения графика записей по месяцам.")
        
        # Тепловая карта по годам и месяцам
        st.subheader("Тепловая карта количества записей по годам и месяцам")
        
        # Добавляем год и месяц как отдельные колонки
        data_heatmap = data.copy()
        data_heatmap['Год'] = data_heatmap['ДатаСоздан'].dt.year
        data_heatmap['Месяц'] = data_heatmap['ДатаСоздан'].dt.month
        
        if not data_heatmap.empty and len(data_heatmap['Год'].unique()) > 0 and len(data_heatmap['Месяц'].unique()) > 0:
            # Группируем по годам и месяцам
            heatmap_data = data_heatmap.groupby(['Год', 'Месяц']).size().reset_index()
            heatmap_data.columns = ['Год', 'Месяц', 'Количество записей']

            # Создаем сводную таблицу
            heatmap_pivot = heatmap_data.pivot_table(
                values='Количество записей', 
                index='Год', 
                columns='Месяц',
                aggfunc='sum',
                fill_value=0 # Заполняем пропуски нулями для тепловой карты
            )
            
            # Названия месяцев
            month_names = {
                1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
                7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
            }
            
            # Создаем тепловую карту
            fig_heatmap_records = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=[month_names.get(m, m) for m in heatmap_pivot.columns],
                y=heatmap_pivot.index,
                colorscale="Viridis",
                colorbar=dict(title="Кол-во записей"),
                hoverongaps=False
            ))
            
            fig_heatmap_records.update_layout(
                title="Тепловая карта количества записей по годам и месяцам",
                xaxis_title="Месяц",
                yaxis_title="Год",
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_heatmap_records, use_container_width=True)
            st.markdown(get_general_explanation("heatmap_monthly_yearly_records"))
        else:
             st.info("Недостаточно данных для построения тепловой карты записей.")
        
        # Сезонный график средних цен по месяцам
        st.subheader("Сезонность средних цен по месяцам")
        if 'Цена нетто' in data.columns and pd.api.types.is_numeric_dtype(data['Цена нетто']):
            price_data = data.dropna(subset=['Цена нетто']).copy()
            price_data['Год'] = price_data['ДатаСоздан'].dt.year
            price_data['Месяц'] = price_data['ДатаСоздан'].dt.month

            monthly_prices = price_data.groupby(['Год', 'Месяц'])['Цена нетто'].mean().reset_index()
            
            if not monthly_prices.empty:
                # Добавляем названия месяцев
                monthly_prices['Месяц название'] = monthly_prices['Месяц'].map(month_names)

                # Строим график для каждого года
                fig_season_price = px.line(
                    monthly_prices,
                    x='Месяц',
                    y='Цена нетто',
                    color='Год',
                    markers=True,
                    title='Общая сезонность средних цен по месяцам (по годам)',
                    labels={'Месяц': 'Месяц', 'Цена нетто': 'Средняя цена нетто'}
                )
                
                # Настраиваем оси
                fig_season_price.update_xaxes(
                    tickmode='array',
                    tickvals=list(range(1, 13)),
                    ticktext=[month_names.get(m, m) for m in range(1, 13)]
                )
                fig_season_price.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    autosize=True
                )

                st.plotly_chart(fig_season_price, use_container_width=True)
                st.markdown(get_general_explanation("line_monthly_avg_price"))
            else:
                 st.info("Нет данных для построения графика сезонности цен.")
        else:
             st.warning("Столбец 'Цена нетто' отсутствует или не является числовым. Сезонность цен недоступна.")
    
    def plot_volatility(self, volatility_data):
        """
        Визуализирует волатильность цен материалов
        """
        st.subheader("Визуализация волатильности цен материалов")
        if volatility_data is None or volatility_data.empty:
             st.warning("Нет данных для анализа волатильности.")
             return
        if 'Материал' not in volatility_data.columns or 'Коэффициент вариации' not in volatility_data.columns:
             st.error("В данных волатильности отсутствуют необходимые колонки.")
             return

        # Топ-20 материалов с наибольшей волатильностью
        # Убираем NaN и inf перед сортировкой
        volatility_data_clean = volatility_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Коэффициент вариации'])
        if not volatility_data_clean.empty:
             top_volatile = volatility_data_clean.nlargest(20, 'Коэффициент вариации')

             fig_top_vol = px.bar(
                 top_volatile,
                 x='Материал',
                 y='Коэффициент вариации',
                 text='Коэффициент вариации',
                 title='Топ-20 материалов с наибольшей волатильностью цен (%)'
             )
             # Отображаем как проценты
             fig_top_vol.update_traces(texttemplate='%{text:.1f}', textposition='outside')
             fig_top_vol.update_layout(
                 xaxis_tickangle=-45,
                 height=500,
                 margin=dict(l=20, r=20, t=40, b=20),
                 yaxis_title="Коэффициент вариации, %",
                 autosize=True
             )

             st.plotly_chart(fig_top_vol, use_container_width=True)
             st.markdown(get_general_explanation("bar_top_volatile"))
        else:
             st.info("Нет данных для отображения топ волатильных материалов.")
        
        # Топ-20 материалов с наименьшей волатильностью (но не нулевой)
        bottom_volatile_data = volatility_data_clean[volatility_data_clean['Коэффициент вариации'] > 0]
        if not bottom_volatile_data.empty:
             bottom_volatile = bottom_volatile_data.nsmallest(20, 'Коэффициент вариации')

             fig_bot_vol = px.bar(
                 bottom_volatile,
                 x='Материал',
                 y='Коэффициент вариации',
                 text='Коэффициент вариации',
                 title='Топ-20 материалов с наименьшей ненулевой волатильностью цен (%)'
             )
             fig_bot_vol.update_traces(texttemplate='%{text:.1f}', textposition='outside')
             fig_bot_vol.update_layout(
                 xaxis_tickangle=-45,
                 height=500,
                 margin=dict(l=20, r=20, t=40, b=20),
                 yaxis_title="Коэффициент вариации, %",
                 autosize=True
             )

             st.plotly_chart(fig_bot_vol, use_container_width=True)
             st.markdown(get_general_explanation("bar_bottom_volatile"))
        else:
             st.info("Нет материалов с ненулевой волатильностью для отображения.")
        
        # Распределение коэффициентов вариации
        st.subheader("Распределение коэффициентов вариации")
        
        # Ограничиваем до 100% для наглядности
        histogram_data = volatility_data_clean[volatility_data_clean['Коэффициент вариации'] <= 100]
        if not histogram_data.empty:
             max_cv_hist = int(histogram_data['Коэффициент вариации'].max())
             if max_cv_hist < 1:
                  max_cv_hist = 1

             fig_cv_hist = px.histogram(
                 histogram_data,
                 x='Коэффициент вариации',
                 nbins=min(50, max_cv_hist),
                 title='Распределение Коэффициентов Вариации (до 100%)'
             )
             fig_cv_hist.update_layout(
                 height=500,
                 margin=dict(l=20, r=20, t=40, b=20),
                 xaxis_title="Коэффициент вариации, %",
                 yaxis_title="Количество материалов",
                 autosize=True
             )

             st.plotly_chart(fig_cv_hist, use_container_width=True)
             st.markdown(get_general_explanation("hist_cv_distribution"))
        else:
             st.info("Нет данных для построения гистограммы волатильности (<=100%).")
        
        # Категоризация материалов по волатильности
        st.subheader("Категоризация материалов по волатильности")
        
        if not volatility_data_clean.empty:
            # Создаем категории
            bins = [-np.inf, 5, 10, 20, 30, 50, 100, np.inf] # Начинаем с -inf чтобы включить 0
            labels = ["<5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]

            try:
                volatility_data_clean['Категория волатильности'] = pd.cut(
                    volatility_data_clean['Коэффициент вариации'],
                    bins=bins,
                    labels=labels,
                    right=False # Левая граница включается
                )
                volatility_data_clean['Категория волатильности'] = volatility_data_clean['Категория волатильности'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                category_counts = volatility_data_clean.groupby('Категория волатильности', observed=True)['Материал'].count().reset_index()
                category_counts.columns = ['Категория волатильности', 'Количество материалов']
                category_counts = category_counts[category_counts['Количество материалов'] > 0]

                if not category_counts.empty:
                    fig_cv_pie = px.pie(
                        category_counts,
                        names='Категория волатильности',
                        values='Количество материалов',
                        title='Распределение материалов по категориям волатильности'
                    )
                    fig_cv_pie.update_traces(textinfo='percent+label')
                    fig_cv_pie.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        autosize=True
                    )

                    st.plotly_chart(fig_cv_pie, use_container_width=True)
                    st.markdown(get_general_explanation("pie_cv_categories"))
                else:
                     st.info("Нет данных для построения диаграммы категорий волатильности.")
            except Exception as e:
                 st.error(f"Ошибка при категоризации волатильности: {e}")
        else:
             st.info("Нет очищенных данных для категоризации волатильности.")
    
    def plot_stability(self, stability_data):
        """
        Визуализирует стабильность цен материалов
        """
        st.subheader("Визуализация стабильных цен материалов")
        if stability_data is None or stability_data.empty:
             st.warning("Нет данных для анализа стабильности.")
             return
        required_cols = ['Материал', 'Стабильная цена', 'Количество записей', 'Процент одинаковых значений']
        if not all(col in stability_data.columns for col in required_cols):
             st.error("В данных стабильности отсутствуют необходимые колонки.")
             return

        # Количество материалов со стабильными и нестабильными ценами
        stability_counts = stability_data['Стабильная цена'].value_counts().reset_index()
        stability_counts.columns = ['Стабильная цена', 'Количество материалов']

        if not stability_counts.empty:
            # Заменяем булевы значения на текст
            stability_counts['Стабильная цена'] = stability_counts['Стабильная цена'].map({
                True: 'Стабильная цена (≥80%)',
                False: 'Нестабильная цена (<80%)'
            }).fillna('Неизвестно')

            fig_stab_pie = px.pie(
                stability_counts,
                names='Стабильная цена',
                values='Количество материалов',
                title='Распределение материалов по стабильности цен'
            )
            fig_stab_pie.update_traces(textinfo='percent+label')
            fig_stab_pie.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_stab_pie, use_container_width=True)
            st.markdown(get_general_explanation("pie_stability_distribution"))
        else:
             st.info("Нет данных для построения диаграммы стабильности цен.")

        # Топ-20 материалов с наибольшим процентом одинаковых значений
        st.subheader("Топ материалов с наибольшим процентом одинаковых значений")

        # Фильтруем только материалы с несколькими записями и валидным процентом
        stable_data_filtered = stability_data[
             (stability_data['Количество записей'] > 1) & 
             (stability_data['Процент одинаковых значений'].notna())]
             
        if not stable_data_filtered.empty:
             top_stable_perc = stable_data_filtered.nlargest(20, 'Процент одинаковых значений')

             fig_top_stab = px.bar(
                 top_stable_perc,
                 x='Материал',
                 y='Процент одинаковых значений',
                 text='Процент одинаковых значений',
                 title='Топ-20 материалов с наибольшим % одинаковых значений цены'
             )
             fig_top_stab.update_traces(texttemplate='%{text:.1f}', textposition='outside')
             fig_top_stab.update_layout(
                 xaxis_tickangle=-45,
                 height=500,
                 margin=dict(l=20, r=20, t=40, b=20),
                 yaxis_title="Процент одинаковых значений, %",
                 autosize=True
             )

             st.plotly_chart(fig_top_stab, use_container_width=True)
             st.markdown(get_general_explanation("bar_top_stable_percentage"))
        else:
             st.info("Нет данных для отображения топ стабильных материалов.")

        # Соотношение между количеством записей и стабильностью цен
        st.subheader("Соотношение между количеством записей и стабильностью цен")

        # Ограничиваем количество записей для наглядности и берем валидные данные
        scatter_data = stability_data[
            (stability_data['Количество записей'] <= 100) & 
            (stability_data['Процент одинаковых значений'].notna())].copy()

        if not scatter_data.empty:
            # Добавляем цветовую маркировку для стабильных и нестабильных цен
            scatter_data['Статус'] = scatter_data['Стабильная цена'].map({
                True: 'Стабильная цена (≥80%)',
                False: 'Нестабильная цена (<80%)'
            }).fillna('Неизвестно')

            fig_scatter_stab = px.scatter(
                scatter_data,
                x='Количество записей',
                y='Процент одинаковых значений',
                color='Статус',
                hover_name='Материал',
                title='Стабильность цены vs Количество записей (до 100)',
                labels={'Процент одинаковых значений': 'Процент одинаковых значений цены (%)'}
            )

            # Добавляем горизонтальную линию на уровне 80%
            fig_scatter_stab.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Порог 80%")

            fig_scatter_stab.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_scatter_stab, use_container_width=True)
            st.markdown(get_general_explanation("scatter_records_vs_stability"))
        else:
             st.info("Нет данных для построения графика соотношения стабильности и количества записей.")
    
    def plot_inactivity(self, inactivity_data):
        """
        Визуализирует неактивные материалы
        """
        st.subheader("Визуализация неактивных материалов")
        if inactivity_data is None or inactivity_data.empty:
             st.warning("Нет данных для анализа неактивности.")
             return
        required_cols = ['Материал', 'Дней с последней активности', 'Неактивный материал', 'Последняя активность материала']
        if not all(col in inactivity_data.columns for col in required_cols):
             st.error(f"В данных неактивности отсутствуют необходимые колонки: {required_cols}")
             return
        if not pd.api.types.is_datetime64_any_dtype(inactivity_data['Последняя активность материала']):
            st.error("Колонка 'Последняя активность материала' имеет неверный формат.")
            return

        # Создаем категории по времени неактивности
        bins = [0, 30, 90, 180, 365, 730, np.inf]
        labels = ["<30 дн", "30-90 дн", "90-180 дн", "180-365 дн", "1-2 г", ">2 л"]

        # Убираем NaN перед категоризацией
        inactivity_data_clean = inactivity_data.dropna(subset=['Дней с последней активности']).copy()

        if not inactivity_data_clean.empty:
            try:
                inactivity_data_clean['Категория неактивности'] = pd.cut(
                    inactivity_data_clean['Дней с последней активности'],
                    bins=bins,
                    labels=labels,
                    right=False # Интервалы [a, b)
                )
                inactivity_data_clean['Категория неактивности'] = inactivity_data_clean['Категория неактивности'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                category_counts = inactivity_data_clean.groupby('Категория неактивности', observed=True)['Материал'].count().reset_index()
                category_counts.columns = ['Категория неактивности', 'Количество материалов']
                category_counts = category_counts[category_counts['Количество материалов'] > 0]

                # Задаем правильный порядок категорий
                category_order = [l for l in labels if l in category_counts['Категория неактивности'].unique()]
                if 'Неизвестно' in category_counts['Категория неактивности'].unique():
                    category_order.append('Неизвестно')

                if not category_counts.empty:
                    fig_inact_cat = px.bar(
                        category_counts,
                        x='Категория неактивности',
                        y='Количество материалов',
                        text='Количество материалов',
                        title='Распределение материалов по времени неактивности',
                        category_orders={'Категория неактивности': category_order}
                    )
                    fig_inact_cat.update_traces(texttemplate='%{text:,}', textposition='outside')
                    fig_inact_cat.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title="Период с последней активности",
                        yaxis_title="Количество материалов",
                        autosize=True
                    )

                    st.plotly_chart(fig_inact_cat, use_container_width=True)
                    st.markdown(get_general_explanation("bar_inactivity_categories"))
                else:
                    st.info("Нет данных для построения графика категорий неактивности.")

            except Exception as e:
                 st.error(f"Ошибка при категоризации неактивности: {e}")
        else:
             st.info("Нет данных с корректными днями неактивности.")

        # Круговая диаграмма активных и неактивных материалов
        st.subheader("Активные и неактивные материалы")

        activity_counts = inactivity_data['Неактивный материал'].value_counts().reset_index()
        activity_counts.columns = ['Статус', 'Количество материалов']

        if not activity_counts.empty:
            # Заменяем булевы значения на текст
            activity_counts['Статус'] = activity_counts['Статус'].map({
                True: 'Неактивный (>365 дн)',
                False: 'Активный (≤365 дн)'
            }).fillna('Неизвестно')

            fig_act_pie = px.pie(
                activity_counts,
                names='Статус',
                values='Количество материалов',
                title='Распределение материалов по активности (порог 365 дней)'
            )
            fig_act_pie.update_traces(textinfo='percent+label')
            fig_act_pie.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_act_pie, use_container_width=True)
            st.markdown(get_general_explanation("pie_activity_status"))
        else:
             st.info("Нет данных для построения диаграммы активности.")

        # Топ-20 материалов с наибольшим периодом неактивности
        st.subheader("Топ материалов с наибольшим периодом неактивности")

        if not inactivity_data_clean.empty:
            top_inactive = inactivity_data_clean.nlargest(20, 'Дней с последней активности')

            fig_top_inact = px.bar(
                top_inactive,
                x='Материал',
                y='Дней с последней активности',
                text='Дней с последней активности',
                title='Топ-20 материалов по длительности неактивности'
            )
            fig_top_inact.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_top_inact.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Дней с последней активности",
                autosize=True
            )

            st.plotly_chart(fig_top_inact, use_container_width=True)
            st.markdown(get_general_explanation("bar_top_inactive"))
        else:
             st.info("Нет данных для отображения топ неактивных материалов.")

        # График распределения последних дат активности
        st.subheader("Распределение последних дат активности")

        if not inactivity_data.empty:
            # Группируем по месяцам последней активности
            last_activity_data = inactivity_data.set_index('Последняя активность материала')
            # Указываем частоту 'ME'
            last_activity = last_activity_data.resample('ME').size().reset_index()
            last_activity.columns = ['Дата', 'Количество материалов']
            last_activity = last_activity[last_activity['Количество материалов'] > 0]

            if not last_activity.empty:
                fig_last_act = px.bar(
                    last_activity,
                    x='Дата',
                    y='Количество материалов',
                    title='Количество материалов по дате последней активности (по месяцам)'
                )
                fig_last_act.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Месяц последней активности",
                    yaxis_title="Количество материалов",
                    autosize=True
                )

                st.plotly_chart(fig_last_act, use_container_width=True)
                st.markdown(get_general_explanation("bar_last_activity_distribution"))
            else:
                 st.info("Нет данных для построения графика распределения дат последней активности.")
        else:
             st.info("Нет данных о последней активности материалов.")
    
    def plot_segmentation_results(self, segments, stats):
        """
        Визуализирует результаты сегментации материалов
        """
        st.subheader("Результаты сегментации материалов")
        
        if not stats or not segments:
             st.warning("Нет данных для визуализации сегментации.")
             return

        # Общие результаты сегментации
        segment_counts = pd.DataFrame({
            'Сегмент': list(stats.keys()),
            'Количество материалов': list(stats.values())
        })
        segment_counts = segment_counts[segment_counts['Количество материалов'] > 0] # Убираем пустые сегменты

        if not segment_counts.empty:
            fig_seg_bar = px.bar(
                segment_counts,
                x='Сегмент',
                y='Количество материалов',
                text='Количество материалов',
                title='Распределение материалов по сегментам'
            )
            fig_seg_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_seg_bar.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_seg_bar, use_container_width=True)
            # TODO: Add explanation for segmentation bar chart

            # Круговая диаграмма распределения по сегментам
            fig_seg_pie = px.pie(
                segment_counts,
                names='Сегмент',
                values='Количество материалов',
                title='Распределение материалов по сегментам'
            )
            fig_seg_pie.update_traces(textinfo='percent+label')
            fig_seg_pie.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                autosize=True
            )

            st.plotly_chart(fig_seg_pie, use_container_width=True)
            # TODO: Add explanation for segmentation pie chart
        else:
            st.info("Нет материалов, попавших в какие-либо сегменты.")

        # Детализация по сегментам
        st.subheader("Детализация по сегментам")
        
        # Объединяем все сегменты в один DataFrame
        all_segments = pd.DataFrame()

        for segment, data in segments.items():
            if data is not None and not data.empty:
                data_copy = data.copy()
                data_copy['Сегмент'] = segment
                all_segments = pd.concat([all_segments, data_copy], ignore_index=True)

        # Выбор сегмента для просмотра
        available_segments = list(stats.keys())
        segment_selection = st.selectbox(
            "Выберите сегмент для просмотра:",
            ['Все сегменты'] + available_segments,
             key="segment_details_select"
        )

        if segment_selection == 'Все сегменты':
            display_data = all_segments
        elif segment_selection in segments:
            display_data = segments[segment_selection]
        else:
            display_data = pd.DataFrame() # Пустой DataFrame, если сегмент не найден

        # Показываем таблицу с данными выбранного сегмента с пагинацией
        st.write(f"Данные сегмента: {segment_selection}")

        if display_data is not None and not display_data.empty:
            # Проверка на размер данных и добавление пагинации при необходимости
            row_count = len(display_data)

            if row_count > 1000:
                st.info(f"Найдено {row_count:,} строк данных. Отображение с пагинацией.".replace(",", " "))

                # Добавляем пагинацию
                page_size = st.slider("Строк на странице:",
                                    min_value=50,
                                    max_value=500,
                                    value=100,
                                    step=50,
                                     key="segment_page_size")

                total_pages = max(1, (row_count + page_size - 1) // page_size)
                page_number = st.number_input("Страница:",
                                            min_value=1,
                                            max_value=total_pages,
                                            value=1,
                                             key="segment_page_number")

                # Вычисляем индексы для текущей страницы
                start_idx = (page_number - 1) * page_size
                end_idx = min(start_idx + page_size, row_count)

                # Отображаем данные для текущей страницы
                paged_data = display_data.iloc[start_idx:end_idx].copy()

                st.write(f"Отображение строк {start_idx+1:,}-{end_idx:,} из {row_count:,}".replace(",", " "))

                st.dataframe(
                    create_styled_dataframe(
                        paged_data,
                        precision=2
                    ),
                    use_container_width=True
                )

                # Добавляем возможность экспорта всех данных
                csv = display_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Скачать все {row_count:,} строк в CSV".replace(",", " "),
                    data=csv,
                    file_name=f"segment_{segment_selection.replace(' ', '_')}_all.csv",
                    mime="text/csv",
                    key='download_segment_all_data'
                )
            else:
                st.dataframe(
                    create_styled_dataframe(
                        display_data,
                        precision=2
                    ),
                    use_container_width=True
                )
        else:
            st.info(f"В сегменте '{segment_selection}' нет данных для отображения.")

        # Визуализация характеристик сегментов
        if not all_segments.empty:
            st.subheader("Сравнение характеристик сегментов")

            # Проверяем наличие необходимых колонок
            numeric_cols_for_comparison = [
                 col for col in [
                     'Количество записей материала', 
                     'Коэффициент вариации цены', 
                     'Временной диапазон материала'
                 ] if col in all_segments.columns and pd.api.types.is_numeric_dtype(all_segments[col])
            ]

            if numeric_cols_for_comparison:
                 # Расчет средних значений по сегментам
                 segment_avg_stats = all_segments.groupby('Сегмент')[numeric_cols_for_comparison].mean().reset_index()

                 # Создаем подграфики
                 if len(numeric_cols_for_comparison) > 0:
                      fig_comp = make_subplots(
                           rows=1, 
                           cols=len(numeric_cols_for_comparison), 
                           subplot_titles=numeric_cols_for_comparison
                      )

                      col_map = {
                           'Количество записей материала': 'Ср. кол-во записей',
                           'Коэффициент вариации цены': 'Ср. КВ цены, %',
                           'Временной диапазон материала': 'Ср. диапазон, дни'
                      }

                      for i, col in enumerate(numeric_cols_for_comparison):
                           # Используем bar chart для наглядности сравнения средних
                           fig_comp.add_trace(
                                go.Bar(
                                     x=segment_avg_stats['Сегмент'], 
                                     y=segment_avg_stats[col],
                                     name=col_map.get(col, col),
                                     text=segment_avg_stats[col].apply(lambda x: f"{x:.1f}"),
                                     textposition='auto'
                                ), 
                                row=1, 
                                col=i+1
                           )
                           fig_comp.update_yaxes(title_text=col_map.get(col, col), row=1, col=i+1)
                      
                      fig_comp.update_layout(
                           title_text="Средние характеристики материалов по сегментам", 
                           height=500,
                           showlegend=False,
                           margin=dict(l=40, r=20, t=60, b=20)
                      )
                      st.plotly_chart(fig_comp, use_container_width=True)
                      # TODO: Add explanation for segment comparison chart
            else:
                 st.info("Недостаточно числовых колонок для сравнения характеристик сегментов.")
        else:
             st.info("Нет объединенных данных по сегментам для сравнения.")