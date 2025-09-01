import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

class MaterialSegmenter:
    """
    Класс для сегментации временных рядов на основе различных критериев.
    Строит метрики волатильности, стабильности и неактивности.
    Требует передачи role-based имён колонок через role_names и работает
    только с каноническими role-именами (не использует legacy fallback).
    """

    def __init__(self, role_names=None):
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

    # --- Helpers to resolve columns ---
    def _resolve_material_col(self, df: pd.DataFrame):
        # Only accept the canonical ROLE_ID
        return self.ROLE_ID if self.ROLE_ID in df.columns else None

    def _resolve_date_col(self, df: pd.DataFrame):
        # Only accept the canonical ROLE_DATE
        return self.ROLE_DATE if self.ROLE_DATE in df.columns else None

    def _resolve_price_col(self, df: pd.DataFrame):
        # prefer role-based normalized target column
        norm_col = f"{self.ROLE_TARGET} (норм.)"
        # Only accept role-based normalized price or role target
        if norm_col in df.columns:
            return norm_col
        if self.ROLE_TARGET in df.columns:
            return self.ROLE_TARGET
        return None

    # --- Public analysis entry points ---
    def analyze_volatility(self, data: pd.DataFrame):
        st.header("Анализ волатильности целевых значений")
        st.write("Вычисление коэффициента вариации (CV) для каждого временного ряда.")

        volatility_df = self._calculate_volatility(data)

        # save to session
        st.session_state['volatility_data'] = volatility_df

        # show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Средняя волатильность", f"{volatility_df['Коэффициент вариации'].mean():.2f}%")
        with col2:
            st.metric("Медианная волатильность", f"{volatility_df['Коэффициент вариации'].median():.2f}%")
        with col3:
            low_vol = (volatility_df['Коэффициент вариации'] < 10).sum()
            st.metric("Временные ряды с низкой волатильностью", f"{low_vol / len(volatility_df) * 100:.1f}%")

        # show tables
        from modules.utils import format_streamlit_dataframe
        st.subheader("Топ-20 по волатильности")
        st.dataframe(format_streamlit_dataframe(volatility_df.head(20)), use_container_width=True, height=400)

        st.subheader("Топ-20 с наименьшей ненулевой волатильностью")
        st.dataframe(format_streamlit_dataframe(volatility_df[volatility_df['Коэффициент вариации'] > 0].tail(20)), use_container_width=True, height=400)

        return volatility_df

    def analyze_stability(self, data: pd.DataFrame):
        st.header("Анализ стабильности целевых значений")
        stability_df = self._calculate_stability(data)
        st.session_state['stability_data'] = stability_df

        from modules.utils import format_streamlit_dataframe
        st.dataframe(format_streamlit_dataframe(stability_df.head(50)), use_container_width=True, height=400)
        return stability_df

    def analyze_inactivity(self, data: pd.DataFrame):
        st.header("Анализ неактивности временных рядов")
        inactivity_df = self._calculate_inactivity(data)
        st.session_state['inactivity_data'] = inactivity_df

        from modules.utils import format_streamlit_dataframe
        st.dataframe(format_streamlit_dataframe(inactivity_df.head(50)), use_container_width=True, height=400)
        return inactivity_df

    # --- Core calculators ---
    def _calculate_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        material_col = self._resolve_material_col(df)
        price_col = self._resolve_price_col(df)

        if material_col is None:
            raise KeyError(f"Не найден идентификатор временного ряда. Ожидаемая колонка: '{self.ROLE_ID}'")
        if price_col is None:
            raise KeyError(f"Не найдена целевая колонка. Ожидалась '{self.ROLE_TARGET} (норм.)' или '{self.ROLE_TARGET}'")

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

        grouped = df.groupby(material_col)[price_col].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        grouped = grouped.rename(columns={'count': 'Количество записей', 'mean': 'Среднее значение', 'std': 'Стандартное отклонение', 'min': 'Минимальное значение', 'max': 'Максимальное значение'})
        grouped['Коэффициент вариации'] = (grouped['Стандартное отклонение'] / grouped['Среднее значение'].replace(0, np.nan)) * 100
        grouped['Коэффициент вариации'] = grouped['Коэффициент вариации'].fillna(0)
        grouped['Диапазон значений'] = grouped['Максимальное значение'] - grouped['Минимальное значение']
        grouped['Процентное изменение'] = np.where(grouped['Минимальное значение'] != 0, (grouped['Диапазон значений'] / grouped['Минимальное значение']) * 100, 0)

        grouped = grouped.sort_values('Коэффициент вариации', ascending=False).reset_index(drop=True)
        # ensure identifier column name equals role-based ID
        if material_col != self.ROLE_ID and self.ROLE_ID not in grouped.columns:
            grouped = grouped.rename(columns={material_col: self.ROLE_ID})
        return grouped

    def _calculate_stability(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        material_col = self._resolve_material_col(df)
        price_col = self._resolve_price_col(df)

        if material_col is None:
            raise KeyError(f"Не найден идентификатор временного ряда. Ожидаемая колонка: '{self.ROLE_ID}'")
        if price_col is None:
            raise KeyError(f"Не найдена целевая колонка. Ожидалась '{self.ROLE_TARGET} (норм.)' или '{self.ROLE_TARGET}'")

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

        results = []
        for material, grp in df.groupby(material_col):
            price_counts = grp[price_col].value_counts(dropna=True)
            num_records = len(grp)
            most_common_price = price_counts.index[0] if len(price_counts) > 0 else np.nan
            most_common_count = price_counts.iloc[0] if len(price_counts) > 0 else 0
            percent_same = (most_common_count / num_records) * 100 if num_records > 0 else 0
            is_stable = percent_same >= 80
            results.append({
                    self.ROLE_ID: material,
                'Количество записей': num_records,
                'Наиболее частое значение': most_common_price,
                'Количество одинаковых значений': most_common_count,
                'Процент одинаковых значений': percent_same,
                'Стабильное значение': is_stable,
                'Количество уникальных значений': int(price_counts.size)
            })

        stability_df = pd.DataFrame(results).sort_values('Процент одинаковых значений', ascending=False).reset_index(drop=True)
        # rename identifier column to canonical role id
        if material_col != self.ROLE_ID and self.ROLE_ID not in stability_df.columns:
            stability_df = stability_df.rename(columns={material_col: self.ROLE_ID})
        return stability_df

    def _calculate_inactivity(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        material_col = self._resolve_material_col(df)
        date_col = self._resolve_date_col(df)

        if material_col is None:
            raise KeyError(f"Не найден идентификатор временного ряда. Ожидаемая колонка: '{self.ROLE_ID}'")
        if date_col is None:
            raise KeyError(f"Не найдена колонка с датой. Ожидаемая колонка: '{self.ROLE_DATE}'")

        # ensure datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        current_date = df[date_col].max()

        results = []
        for material, grp in df.groupby(material_col):
            num_records = len(grp)
            first_date = grp[date_col].min()
            last_date = grp[date_col].max()
            days_span = (last_date - first_date).days if num_records > 1 else 0
            avg_interval = days_span / (num_records - 1) if num_records > 1 else None
            days_since_last = (current_date - last_date).days if pd.notnull(last_date) and pd.notnull(current_date) else None
            is_inactive = days_since_last is not None and days_since_last > 365

            results.append({
                    self.ROLE_ID: material,
                'Количество записей': num_records,
                'Первая дата': first_date,
                'Последняя дата': last_date,
                'Временной диапазон (дни)': days_span,
                'Средний интервал (дни)': avg_interval,
                'Последняя активность': last_date,
                'Дней с последней активности': days_since_last,
                'Неактивный временной ряд': is_inactive
            })

        inactivity_df = pd.DataFrame(results).sort_values('Дней с последней активности', ascending=False).reset_index(drop=True)
        # rename identifier column to canonical role id
        if material_col != self.ROLE_ID and self.ROLE_ID not in inactivity_df.columns:
            inactivity_df = inactivity_df.rename(columns={material_col: self.ROLE_ID})
        return inactivity_df