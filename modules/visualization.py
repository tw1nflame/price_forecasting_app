# filepath: c:\price_forecasting_app\modules\visualizer.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.utils import create_styled_dataframe, get_general_explanation
import io

class Visualizer:
    """
    Класс для визуализации данных о временных рядах
    """
    
    def __init__(self, role_names=None):
        # role_names: dict with ROLE_ID, ROLE_DATE, ROLE_TARGET, ROLE_QTY, ROLE_CURRENCY, ROLE_RATE
        self.role_names = role_names or {
            'ROLE_ID': 'ID',
            'ROLE_DATE': 'Дата',
            'ROLE_TARGET': 'Целевая Колонка',
            'ROLE_QTY': 'Количество',
            'ROLE_CURRENCY': 'Валюта',
            'ROLE_RATE': 'Курс'
        }
        self.ROLE_ID = self.role_names.get('ROLE_ID')
        self.ROLE_DATE = self.role_names.get('ROLE_DATE')
        self.ROLE_TARGET = self.role_names.get('ROLE_TARGET')
        self.ROLE_QTY = self.role_names.get('ROLE_QTY')
        self.ROLE_CURRENCY = self.role_names.get('ROLE_CURRENCY')
        self.ROLE_RATE = self.role_names.get('ROLE_RATE')
    
    def plot_materials_distribution(self, data):
        """
        Визуализирует распределение временных рядов
        """
        st.subheader("Визуализация распределения временных рядов")
        if self.ROLE_ID not in data.columns or self.ROLE_DATE not in data.columns:
            st.error(f"Отсутствуют необходимые колонки '{self.ROLE_ID}' или '{self.ROLE_DATE}'.")
            return

        # Распределение временных рядов по количеству данных
        materials_count = data.groupby(self.ROLE_ID)[self.ROLE_DATE].count().reset_index()
        materials_count.columns = [self.ROLE_ID, 'Количество данных']
        materials_count = materials_count.sort_values('Количество данных', ascending=False)

        # Отображаем топ-20 временных рядов
        if not materials_count.empty:
            top_materials = materials_count.head(20)

            fig_top_mat = px.bar(
                top_materials,
                x=self.ROLE_ID,
                y='Количество данных',
                text='Количество данных',
                title='Топ-20 временных рядов по количеству данных',
                hover_data={
                    self.ROLE_ID: True,
                    'Количество данных': ':,',
                }
            )
            fig_top_mat.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_top_mat.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_tickformat=',',
                autosize=True
            )

            st.plotly_chart(fig_top_mat, use_container_width=True)
            st.markdown(get_general_explanation("bar_top_materials_records"))
        else:
            st.info("Нет данных для отображения топ временных рядов.")

        # Распределение по группам (если есть)
        group_col = 'ГруппаМтр' if 'ГруппаМтр' in data.columns else None

        if group_col is not None:
            group_counts = data.groupby(group_col)[self.ROLE_ID].nunique().reset_index()
            group_counts.columns = ['Группа', 'Количество уникальных временных рядов']
            group_counts = group_counts.sort_values('Количество уникальных временных рядов', ascending=False)

            if not group_counts.empty:
                top_groups = group_counts.head(10)

                fig_top_grp = px.bar(
                    top_groups,
                    x='Группа',
                    y='Количество уникальных временных рядов',
                    text='Количество уникальных временных рядов',
                    title='Топ-10 групп по количеству уникальных временных рядов'
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
                st.info("Нет данных о группах временных рядов.")
        else:
            st.info("Столбец 'ГруппаМтр' отсутствует, распределение по группам недоступно.")

        # Круговая диаграмма распределения по количеству данных
        st.subheader("Распределение временных рядов по количеству данных")

        if not materials_count.empty:
            # Создаем категории
            if materials_count['Количество данных'].max() > 0:
                bins = [0, 1, 5, 10, 20, 50, 100, np.inf]
                labels = ["1", "2-5", "6-10", "11-20",
                         "21-50", "51-100", ">100"]

                try:
                    materials_count['Категория'] = pd.cut(
                        materials_count['Количество данных'],
                        bins=bins,
                        labels=labels,
                        right=True,
                        include_lowest=True
                    )
                    materials_count['Категория'] = materials_count['Категория'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                    category_counts = materials_count.groupby('Категория', observed=True)[self.ROLE_ID].count().reset_index()
                    category_counts.columns = ['Категория', 'Количество временных рядов']
                    category_counts = category_counts[category_counts['Количество временных рядов'] > 0]

                    if not category_counts.empty:
                        fig_cat_pie = px.pie(
                            category_counts,
                            names='Категория',
                            values='Количество временных рядов',
                            title='Доля временных рядов по категориям количества записей'
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
                st.info("Нет временных рядов с >0 записей для категоризации.")
        else:
            st.info("Нет данных о количестве записей временных рядов.")
    
    def plot_time_distribution(self, data):
        """
        Визуализирует временное распределение данных
        """
        st.subheader("Визуализация временного распределения")

        # Проверяем наличие и формат колонки с датой
        if self.ROLE_DATE not in data.columns:
            st.error(f"Отсутствует колонка '{self.ROLE_DATE}' для временного анализа.")
            return
        if not pd.api.types.is_datetime64_any_dtype(data[self.ROLE_DATE]):
            try:
                data[self.ROLE_DATE] = pd.to_datetime(data[self.ROLE_DATE])
            except Exception:
                st.error(f"Колонка '{self.ROLE_DATE}' не является датой и не может быть преобразована.")
                return

        # Названия месяцев для отображения
        month_names = {
            1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
            7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
        }

        # Временной ряд общего количества записей по месяцам
        data_ts = data.set_index(self.ROLE_DATE)
        monthly_data = data_ts.resample('ME').size().reset_index()
        monthly_data.columns = ['Дата', 'Количество данных']

        if not monthly_data.empty:
            fig_ts_records = px.line(
                monthly_data,
                x='Дата',
                y='Количество данных',
                title='Общее количество данных по месяцам'
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

        data_heatmap = data.copy()
        data_heatmap['Год'] = data_heatmap[self.ROLE_DATE].dt.year
        data_heatmap['Месяц'] = data_heatmap[self.ROLE_DATE].dt.month

        if not data_heatmap.empty and len(data_heatmap['Год'].unique()) > 0 and len(data_heatmap['Месяц'].unique()) > 0:
            heatmap_data = data_heatmap.groupby(['Год', 'Месяц']).size().reset_index()
            heatmap_data.columns = ['Год', 'Месяц', 'Количество данных']

            heatmap_pivot = heatmap_data.pivot_table(
                values='Количество данных',
                index='Год',
                columns='Месяц',
                aggfunc='sum',
                fill_value=0
            )

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

        # Сезонный график средних значений целевой колонки по месяцам
        st.subheader("Сезонность средних значений целевой колонки по месяцам")
        norm_col = f"{self.ROLE_TARGET} (норм.)"
        if norm_col in data.columns and pd.api.types.is_numeric_dtype(data[norm_col]):
            price_data = data.dropna(subset=[norm_col]).copy()
            price_data['Год'] = price_data[self.ROLE_DATE].dt.year
            price_data['Месяц'] = price_data[self.ROLE_DATE].dt.month

            monthly_prices = price_data.groupby(['Год', 'Месяц'])[norm_col].mean().reset_index()

            if not monthly_prices.empty:
                monthly_prices['Месяц название'] = monthly_prices['Месяц'].map(month_names)

                fig_season_price = px.line(
                    monthly_prices,
                    x='Месяц',
                    y=norm_col,
                    color='Год',
                    markers=True,
                    title='Общая сезонность средних значений по месяцам (по годам)',
                    labels={'Месяц': 'Месяц', norm_col: f'Среднее значение'}
                )

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
                st.info("Нет данных для построения графика сезонности.")
        else:
            st.warning(f"Столбец '{norm_col}' отсутствует или не является числовым. Анализ сезонности недоступен.")
    
    def plot_volatility(self, volatility_data):
        """
        Визуализирует волатильность целевых значений временных рядов
        """
        st.subheader("Визуализация волатильности целевых значений")
        if volatility_data is None or volatility_data.empty:
            st.warning("Нет данных для анализа волатильности.")
            return

        # ожидаем, что в volatility_data присутствует колонка-идентификатор ROLE_ID
        material_col = self.ROLE_ID if self.ROLE_ID in volatility_data.columns else None
        cv_col = next((c for c in ('Коэффициент вариации', 'CV', 'coef_var', 'cv') if c in volatility_data.columns), None)

        # Если передан агрегированный датасет с коэффициентами волатильности, используем его
        if material_col is not None and cv_col is not None:
            df = volatility_data.copy()
            df[cv_col] = pd.to_numeric(df[cv_col], errors='coerce')
            volatility_data_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[cv_col])
        else:
            # Попытка вычислить волатильность из сырых данных (role-based)
            norm_col = f"{self.ROLE_TARGET} (норм.)"
            if self.ROLE_ID in volatility_data.columns and norm_col in volatility_data.columns:
                df_raw = volatility_data[[self.ROLE_ID, norm_col]].copy()
                df_raw[norm_col] = pd.to_numeric(df_raw[norm_col], errors='coerce')
                grouped = df_raw.groupby(self.ROLE_ID)[norm_col].agg(['count', 'mean', 'std']).reset_index()
                grouped = grouped.rename(columns={'count': 'Количество записей', 'mean': 'Среднее значение', 'std': 'std'})
                grouped['Коэффициент вариации'] = (grouped['std'] / grouped['Среднее значение'].replace(0, np.nan)) * 100
                volatility_data_clean = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=['Коэффициент вариации'])
                material_col = self.ROLE_ID
                cv_col = 'Коэффициент вариации'
            else:
                st.error("Входные данные не содержат предрасчитанных метрик волатильности и не содержат колонок ID и нормализованного целевого значения.")
                return

        # Если нет строк после очистки — сообщаем
        if volatility_data_clean is None or volatility_data_clean.empty:
            st.info("Нет данных для отображения волатильности.")
            return

        # Топ-20 временных рядов с наибольшей волатильностью
        top_volatile = volatility_data_clean.nlargest(20, cv_col)

        hover_cfg = {material_col: True, cv_col: ':.2f%'}
        if 'Среднее значение' in volatility_data_clean.columns:
            hover_cfg['Среднее значение'] = ':.2f'
        if 'Количество записей' in volatility_data_clean.columns:
            hover_cfg['Количество записей'] = True

        fig_top_vol = px.bar(
            top_volatile,
            x=material_col,
            y=cv_col,
            text=cv_col,
            title='Топ-20 временных рядов с наибольшей волатильностью (%)',
            hover_data=hover_cfg
        )
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

        # Топ-20 временных рядов с наименьшей ненулевой волатильностью
        bottom_volatile_data = volatility_data_clean[volatility_data_clean[cv_col] > 0]
        if not bottom_volatile_data.empty:
            bottom_volatile = bottom_volatile_data.nsmallest(20, cv_col)

            fig_bot_vol = px.bar(
                bottom_volatile,
                x=material_col,
                y=cv_col,
                text=cv_col,
                title='Топ-20 временных рядов с наименьшей ненулевой волатильностью (%)',
                hover_data=hover_cfg
            )
            fig_bot_vol.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_bot_vol.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Коэффициент вариации, %",
                yaxis_tickformat='.2f',
                autosize=True
            )

            st.plotly_chart(fig_bot_vol, use_container_width=True)
            st.markdown(get_general_explanation("bar_bottom_volatile"))
        else:
            st.info("Нет временных рядов с ненулевой волатильностью для отображения.")

        # Распределение коэффициентов вариации
        st.subheader("Распределение коэффициентов вариации")
        histogram_data = volatility_data_clean[volatility_data_clean[cv_col] <= 100]
        if not histogram_data.empty:
            max_cv_hist = int(histogram_data[cv_col].max())
            if max_cv_hist < 1:
                max_cv_hist = 1

            fig_cv_hist = px.histogram(
                histogram_data,
                x=cv_col,
                nbins=min(50, max_cv_hist),
                title='Распределение Коэффициентов Вариации (до 100%)'
            )
            fig_cv_hist.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Коэффициент вариации, %",
                yaxis_title="Количество временных рядов",
                autosize=True
            )

            st.plotly_chart(fig_cv_hist, use_container_width=True)
            st.markdown(get_general_explanation("hist_cv_distribution"))
        else:
            st.info("Нет данных для построения гистограммы волатильности (<=100%).")

        # Категоризация временных рядов по волатильности
        st.subheader("Категоризация временных рядов по волатильности")
        if not volatility_data_clean.empty:
            bins = [-np.inf, 5, 10, 20, 30, 50, 100, np.inf]
            labels = ["<5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]

            try:
                volatility_data_clean['Категория волатильности'] = pd.cut(
                    volatility_data_clean[cv_col],
                    bins=bins,
                    labels=labels,
                    right=False
                )
                volatility_data_clean['Категория волатильности'] = volatility_data_clean['Категория волатильности'].cat.add_categories('Неизвестно').fillna('Неизвестно')

                id_col_for_group = self.ROLE_ID if self.ROLE_ID in volatility_data_clean.columns else volatility_data_clean.columns[0]
                category_counts = volatility_data_clean.groupby('Категория волатильности', observed=True)[id_col_for_group].count().reset_index()
                category_counts.columns = ['Категория волатильности', 'Количество временных рядов']
                category_counts = category_counts[category_counts['Количество временных рядов'] > 0]

                if not category_counts.empty:
                    fig_cv_pie = px.pie(
                        category_counts,
                        names='Категория волатильности',
                        values='Количество временных рядов',
                        title='Распределение временных рядов по категориям волатильности'
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
        Визуализирует стабильность целевых значений временных рядов
        """
        st.subheader("Визуализация стабильных целевых значений")
        if stability_data is None or stability_data.empty:
            st.warning("Нет данных для анализа стабильности.")
            return
        # use canonical ROLE_ID instead of legacy 'Материал'
        required_cols = [self.ROLE_ID, 'Стабильное значение', 'Количество записей', 'Процент одинаковых значений']
        if not all(col in stability_data.columns for col in required_cols):
            st.error("В данных стабильности отсутствуют необходимые колонки.")
            return

        # Количество временных рядов со стабильными и нестабильными значениями
        stability_counts = stability_data['Стабильное значение'].value_counts().reset_index()
        stability_counts.columns = ['Стабильное значение', 'Количество временных рядов']

        if not stability_counts.empty:
            # Заменяем булевы значения на текст
            stability_counts['Стабильное значение'] = stability_counts['Стабильное значение'].map({
                True: 'Стабильное значение (≥80%)',
                False: 'Нестабильное значение (<80%)'
            }).fillna('Неизвестно')

            fig_stab_pie = px.pie(
                stability_counts,
                names='Стабильное значение',
                values='Количество временных рядов',
                title='Распределение временных рядов по стабильности целевого значения'
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
             st.info("Нет данных для построения диаграммы стабильности.")

        # Топ-20 временных рядов с наибольшим процентом одинаковых значений
        st.subheader("Топ временных рядов с наибольшим процентом одинаковых значений")

        # Фильтруем только ряды с несколькими записями и валидным процентом
        stable_data_filtered = stability_data[
             (stability_data['Количество записей'] > 1) & 
             (stability_data['Процент одинаковых значений'].notna())]
             
        if not stable_data_filtered.empty:
             top_stable_perc = stable_data_filtered.nlargest(20, 'Процент одинаковых значений')

             # X-axis should use canonical identifier column
             idcol = self.ROLE_ID if self.ROLE_ID in top_stable_perc.columns else top_stable_perc.columns[0]
             fig_top_stab = px.bar(
                 top_stable_perc,
                 x=idcol,
                 y='Процент одинаковых значений',
                 text='Процент одинаковых значений',
                 title='Топ-20 временных рядов с наибольшим % одинаковых целевых значений'
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
             st.info("Нет данных для отображения топ стабильных временных рядов.")

        # Соотношение между количеством записей и стабильностью
        st.subheader("Соотношение между количеством записей и стабильностью")

        # Ограничиваем количество записей для наглядности и берем валидные данные
        scatter_data = stability_data[
            (stability_data['Количество записей'] <= 100) & 
            (stability_data['Процент одинаковых значений'].notna())].copy()

        if not scatter_data.empty:
            # Добавляем цветовую маркировку для стабильных и нестабильных значений
            scatter_data['Статус'] = scatter_data['Стабильное значение'].map({
                True: 'Стабильное значение (≥80%)',
                False: 'Нестабильное значение (<80%)'
            }).fillna('Неизвестно')

            idcol = self.ROLE_ID if self.ROLE_ID in scatter_data.columns else scatter_data.columns[0]
            fig_scatter_stab = px.scatter(
                scatter_data,
                x='Количество записей',
                y='Процент одинаковых значений',
                color='Статус',
                hover_name=idcol,
                title='Стабильность целевого значения vs Количество записей (до 100)',
                labels={'Процент одинаковых значений': 'Процент одинаковых целевых значений (%)'}
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
        Визуализирует неактивные временные ряды
        """
        st.subheader("Визуализация неактивных временных рядов")
        if inactivity_data is None or inactivity_data.empty:
            st.warning("Нет данных для анализа неактивности.")
            return

        required_cols = [self.ROLE_ID, 'Дней с последней активности', 'Неактивный временной ряд', 'Последняя активность']
        if not all(col in inactivity_data.columns for col in required_cols):
            st.error(f"В данных неактивности отсутствуют необходимые колонки: {required_cols}")
            return
        if not pd.api.types.is_datetime64_any_dtype(inactivity_data['Последняя активность']):
            st.error("Колонка 'Последняя активность' имеет неверный формат.")
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

                idcol = self.ROLE_ID if self.ROLE_ID in inactivity_data_clean.columns else inactivity_data_clean.columns[0]
                category_counts = inactivity_data_clean.groupby('Категория неактивности', observed=True)[idcol].count().reset_index()
                category_counts.columns = ['Категория неактивности', 'Количество временных рядов']
                category_counts = category_counts[category_counts['Количество временных рядов'] > 0]

                # Задаем правильный порядок категорий
                category_order = [l for l in labels if l in category_counts['Категория неактивности'].unique()]
                if 'Неизвестно' in category_counts['Категория неактивности'].unique():
                    category_order.append('Неизвестно')

                if not category_counts.empty:
                    fig_inact_cat = px.bar(
                        category_counts,
                        x='Категория неактивности',
                        y='Количество временных рядов',
                        text='Количество временных рядов',
                        title='Распределение временных рядов по времени неактивности',
                        category_orders={'Категория неактивности': category_order}
                    )
                    fig_inact_cat.update_traces(texttemplate='%{text:,}', textposition='outside')
                    fig_inact_cat.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title="Период с последней активности",
                        yaxis_title="Количество временных рядов",
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

        # Круговая диаграмма активных и неактивных временных рядов
        st.subheader("Активные и неактивные временные ряды")

        activity_counts = inactivity_data['Неактивный временной ряд'].value_counts().reset_index()
        activity_counts.columns = ['Статус', 'Количество временных рядов']

        if not activity_counts.empty:
            # Заменяем булевы значения на текст
            activity_counts['Статус'] = activity_counts['Статус'].map({
                True: 'Неактивный (>365 дн)',
                False: 'Активный (≤365 дн)'
            }).fillna('Неизвестно')

            fig_act_pie = px.pie(
                activity_counts,
                names='Статус',
                values='Количество временных рядов',
                title='Распределение временных рядов по активности (порог 365 дней)'
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

        # Топ-20 временных рядов с наибольшим периодом неактивности
        st.subheader("Топ временных рядов с наибольшим периодом неактивности")

        if not inactivity_data_clean.empty:
            top_inactive = inactivity_data_clean.nlargest(20, 'Дней с последней активности')

            idcol_top = self.ROLE_ID if self.ROLE_ID in top_inactive.columns else top_inactive.columns[0]

            hover_map = {
                idcol_top: True,
                'Дней с последней активности': ':,.0f'
            }

            fig_top_inact = px.bar(
                top_inactive,
                x=idcol_top,
                y='Дней с последней активности',
                text='Дней с последней активности',
                title='Топ-20 временных рядов по длительности неактивности',
                hover_data=hover_map
            )
            fig_top_inact.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_top_inact.update_layout(
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Дней с последней активности",
                yaxis_tickformat=',', # Формат оси Y с разделителем тысяч
                autosize=True
            )

            st.plotly_chart(fig_top_inact, use_container_width=True)
            st.markdown(get_general_explanation("bar_top_inactive"))
        else:
             st.info("Нет данных для отображения топ неактивных временных рядов.")

        # График распределения последних дат активности
        st.subheader("Распределение последних дат активности")

        if not inactivity_data.empty:
            # Группируем по месяцам последней активности
            last_activity_data = inactivity_data.set_index('Последняя активность')
            # Указываем частоту 'ME'
            last_activity = last_activity_data.resample('ME').size().reset_index()
            last_activity.columns = ['Дата', 'Количество временных рядов']
            last_activity = last_activity[last_activity['Количество временных рядов'] > 0]

            if not last_activity.empty:
                fig_last_act = px.bar(
                    last_activity,
                    x='Дата',
                    y='Количество временных рядов',
                    title='Количество временных рядов по дате последней активности (по месяцам)'
                )
                fig_last_act.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Месяц последней активности",
                    yaxis_title="Количество временных рядов",
                    autosize=True
                )

                st.plotly_chart(fig_last_act, use_container_width=True)
                st.markdown(get_general_explanation("bar_last_activity_distribution"))
            else:
                 st.info("Нет данных для построения графика распределения дат последней активности.")
        else:
             st.info("Нет данных о последней активности.")
    
    def plot_segmentation_results(self, segments, stats):
        """
        Визуализирует результаты сегментации временных рядов
        """
        st.subheader("Результаты сегментации временных рядов")
        
        if not stats or not segments:
             st.warning("Нет данных для визуализации сегментации.")
             return

        # Общие результаты сегментации
        segment_counts = pd.DataFrame({
            'Сегмент': list(stats.keys()),
            'Количество временных рядов': list(stats.values())
        })
        segment_counts = segment_counts[segment_counts['Количество временных рядов'] > 0] # Убираем пустые сегменты

        if not segment_counts.empty:
            fig_seg_bar = px.bar(
                segment_counts,
                x='Сегмент',
                y='Количество временных рядов',
                text='Количество временных рядов',
                title='Распределение временных рядов по сегментам'
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
                values='Количество временных рядов',
                title='Распределение временных рядов по сегментам'
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
            st.info("Нет временных рядов, попавших в какие-либо сегменты.")

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
             key="vis_seg_details_select"
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
                                     key="vis_seg_page_size")

                total_pages = max(1, (row_count + page_size - 1) // page_size)
                page_number = st.number_input("Страница:",
                                            min_value=1,
                                            max_value=total_pages,
                                            value=1,
                                             key="vis_seg_page_number")

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
                # Используем BytesIO и кодировку utf-8-sig для корректного отображения в Excel
                csv_buffer = io.BytesIO()
                display_data.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label=f"Скачать все {row_count:,} строк в CSV".replace(",", " "),
                    data=csv_data, # Используем данные из буфера
                    file_name=f"segment_{segment_selection.replace(' ', '_')}_all.csv",
                    mime="text/csv; charset=utf-8-sig", # Указываем кодировку в mime-типе
                    key='vis_seg_download_all_data'
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
                     'Количество записей', 
                     'Коэффициент вариации', 
                     'Временной диапазон'
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
                           'Количество записей': 'Ср. кол-во записей',
                           'Коэффициент вариации': 'Ср. КВ, %',
                           'Временной диапазон': 'Ср. диапазон, дни'
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
                           title_text="Средние характеристики временных рядов по сегментам", 
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