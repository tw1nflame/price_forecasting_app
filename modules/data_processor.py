import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    """
    Класс для предварительной обработки данных
    """
    
    def __init__(self):
        pass
    
    def process_data(self, data):
        """
        Выполняет предварительную обработку данных
        
        Args:
            data: pandas DataFrame с исходными данными
        
        Returns:
            pandas DataFrame с обработанными данными
        """
        # Создаем копию данных, чтобы не изменять исходный DataFrame
        df = data.copy()
        
        # Обработка колонки Материал
        df = self._process_material_column(df)
        
        # Обработка колонки ДатаСоздан
        df = self._process_date_column(df)
        
        # Обработка колонки Цена нетто
        df = self._process_price_column(df)
        
        # Обработка колонки Курс
        df = self._process_exchange_rate_column(df)
        
        # Нормализация цен в единую валюту (если несколько валют)
        df = self._normalize_prices(df)
        
        # Заполнение пропущенных значений
        df = self._fill_missing_values(df)
        
        # Добавление дополнительных признаков
        df = self._add_features(df)
        
        return df
    
    def _process_material_column(self, df):
        """
        Обрабатывает колонку Материал
        """
        # Убедимся, что колонка Материал существует
        if "Материал" not in df.columns:
            st.error("Колонка 'Материал' не найдена в данных")
            return df
        
        # Преобразуем в строку, если это не строка
        df["Материал"] = df["Материал"].astype(str)
        
        # Удаляем лишние пробелы
        df["Материал"] = df["Материал"].str.strip()
        
        return df
    
    def _process_date_column(self, df):
        """
        Обрабатывает колонку ДатаСоздан
        """
        # Убедимся, что колонка ДатаСоздан существует
        if "ДатаСоздан" not in df.columns:
            st.error("Колонка 'ДатаСоздан' не найдена в данных")
            return df
        
        # Проверяем формат даты и преобразуем в datetime
        try:
            # Пробуем различные форматы даты
            date_formats = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]
            
            for date_format in date_formats:
                try:
                    df["ДатаСоздан"] = pd.to_datetime(df["ДатаСоздан"], format=date_format)
                    break
                except:
                    continue
            
            # Если ни один формат не подошел, пробуем автоопределение
            if not pd.api.types.is_datetime64_dtype(df["ДатаСоздан"]):
                df["ДатаСоздан"] = pd.to_datetime(df["ДатаСоздан"], errors="coerce")
            
            # Если после всех попыток есть NaT, предупреждаем пользователя
            if df["ДатаСоздан"].isna().any():
                st.warning(f"В колонке 'ДатаСоздан' обнаружены некорректные значения. "
                           f"Количество пропущенных значений: {df['ДатаСоздан'].isna().sum()}")
        
        except Exception as e:
            st.error(f"Ошибка при обработке колонки 'ДатаСоздан': {str(e)}")
        
        return df
    
    def _process_price_column(self, df):
        """
        Обрабатывает колонку Цена нетто
        """
        # Убедимся, что колонка Цена нетто существует
        if "Цена нетто" not in df.columns:
            st.error("Колонка 'Цена нетто' не найдена в данных")
            return df
        
        try:
            # Преобразуем в строку для обработки
            df["Цена нетто"] = df["Цена нетто"].astype(str)
            
            # Убираем пробелы и заменяем запятые на точки
            df["Цена нетто"] = df["Цена нетто"].str.replace(" ", "")
            df["Цена нетто"] = df["Цена нетто"].str.replace(",", ".")
            
            # Преобразуем в числовой формат
            df["Цена нетто"] = pd.to_numeric(df["Цена нетто"], errors="coerce")
            
            # Проверяем на отрицательные значения
            if (df["Цена нетто"] < 0).any():
                st.warning(f"В колонке 'Цена нетто' обнаружены отрицательные значения. "
                           f"Количество отрицательных значений: {(df['Цена нетто'] < 0).sum()}")
            
            # Проверяем на пропущенные значения
            if df["Цена нетто"].isna().any():
                st.warning(f"В колонке 'Цена нетто' обнаружены пропущенные значения. "
                           f"Количество пропущенных значений: {df['Цена нетто'].isna().sum()}")
        
        except Exception as e:
            st.error(f"Ошибка при обработке колонки 'Цена нетто': {str(e)}")
        
        return df
    
    def _process_exchange_rate_column(self, df):
        """
        Обрабатывает колонку Курс
        """
        # Убедимся, что колонка Курс существует
        if "Курс" not in df.columns:
            st.error("Колонка 'Курс' не найдена в данных")
            return df
        
        try:
            # Преобразуем в строку для обработки
            df["Курс"] = df["Курс"].astype(str)
            
            # Убираем пробелы и заменяем запятые на точки
            df["Курс"] = df["Курс"].str.replace(" ", "")
            df["Курс"] = df["Курс"].str.replace(",", ".")
            
            # Преобразуем в числовой формат
            df["Курс"] = pd.to_numeric(df["Курс"], errors="coerce")
            
            # Заполняем пропущенные значения 1 (предполагаем, что это базовая валюта)
            df["Курс"] = df["Курс"].fillna(1.0)
            
            # Проверяем на нулевые или отрицательные значения
            if (df["Курс"] <= 0).any():
                st.warning(f"В колонке 'Курс' обнаружены нулевые или отрицательные значения. "
                          f"Эти значения будут заменены на 1.")
                df.loc[df["Курс"] <= 0, "Курс"] = 1.0
        
        except Exception as e:
            st.error(f"Ошибка при обработке колонки 'Курс': {str(e)}")
        
        return df
    
    def _normalize_prices(self, df):
        """
        Нормализует цены в единую валюту
        """
        # Убедимся, что необходимые колонки существуют
        required_columns = ["Цена нетто", "Курс", "Влт"]
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            st.error(f"Отсутствуют колонки для нормализации цен: {', '.join(missing_columns)}")
            return df
        
        try:
            # Создаем колонку для нормализованной цены
            df["Цена нетто (норм.)"] = df["Цена нетто"] * df["Курс"]
            
            # Определяем базовую валюту (наиболее часто встречающуюся)
            if "Влт" in df.columns:
                base_currency = df["Влт"].mode().iloc[0]
                df["Базовая валюта"] = base_currency
                st.info(f"Базовая валюта для нормализации цен: {base_currency}")
            else:
                df["Базовая валюта"] = "RUB"  # Предполагаем, что базовая валюта - рубли
        
        except Exception as e:
            st.error(f"Ошибка при нормализации цен: {str(e)}")
        
        return df
    
    def _fill_missing_values(self, df):
        """
        Заполняет пропущенные значения
        """
        # Проверяем, есть ли пропущенные значения
        if df.isna().any().any():
            st.warning("В данных обнаружены пропущенные значения")
            
            # Заполняем пропущенные значения в числовых колонках медианой
            numeric_columns = df.select_dtypes(include=["number"]).columns
            for col in numeric_columns:
                if df[col].isna().any():
                    # Заполняем медианой по каждому материалу
                    df[col] = df.groupby("Материал")[col].transform(
                        lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
                    )
            
            # Проверяем, остались ли пропущенные значения
            if df.isna().any().any():
                st.warning("Некоторые пропущенные значения остались после заполнения")
        
        return df
    
    def _add_features(self, df):
        """
        Добавляет дополнительные признаки
        """
        try:
            # Добавляем год, месяц и день
            df["Год"] = df["ДатаСоздан"].dt.year
            df["Месяц"] = df["ДатаСоздан"].dt.month
            df["День"] = df["ДатаСоздан"].dt.day
            
            # Добавляем квартал
            df["Квартал"] = df["ДатаСоздан"].dt.quarter
            
            # Добавляем день недели
            df["День недели"] = df["ДатаСоздан"].dt.dayofweek
            
            # Добавляем количество дней от начала данных
            min_date = df["ДатаСоздан"].min()
            df["Дней от начала"] = (df["ДатаСоздан"] - min_date).dt.days
            
            # Добавляем признак сезонности (зима, весна, лето, осень)
            season_map = {
                1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
                6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень",
                11: "Осень", 12: "Зима"
            }
            df["Сезон"] = df["Месяц"].map(season_map)
            
            # Добавляем количество записей для каждого материала
            df["Количество записей материала"] = df.groupby("Материал")["Материал"].transform("count")
            
            # Добавляем среднюю цену для каждого материала
            df["Средняя цена материала"] = df.groupby("Материал")["Цена нетто"].transform("mean")
            
            # Добавляем стандартное отклонение цены для каждого материала
            df["Стд. отклонение цены материала"] = df.groupby("Материал")["Цена нетто"].transform("std")
            
            # Добавляем коэффициент вариации цены для каждого материала
            df["Коэффициент вариации цены"] = (df["Стд. отклонение цены материала"] / df["Средняя цена материала"]) * 100
            df["Коэффициент вариации цены"] = df["Коэффициент вариации цены"].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Добавляем флаг стабильности цены (80% одинаковых значений)
            def is_stable_price(group):
                # Вычисляем долю наиболее часто встречающегося значения
                most_common_value_count = group.value_counts().iloc[0] if len(group.value_counts()) > 0 else 0
                return most_common_value_count / len(group) >= 0.8
            
            # Применяем функцию ко всем группам материалов
            stable_materials = df.groupby("Материал")["Цена нетто"].apply(is_stable_price)
            
            # Добавляем результат в DataFrame
            df["Стабильная цена"] = df["Материал"].map(stable_materials)
            
            # Добавляем временной диапазон для каждого материала (в днях)
            df["Временной диапазон материала"] = df.groupby("Материал")["ДатаСоздан"].transform(
                lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
            )
            
            # Добавляем время с последней активности (в днях)
            latest_date = df["ДатаСоздан"].max()
            df["Последняя активность материала"] = df.groupby("Материал")["ДатаСоздан"].transform("max")
            df["Дней с последней активности"] = (latest_date - df["Последняя активность материала"]).dt.days
            
            # Добавляем флаг неактивных материалов (более года без активности)
            df["Неактивный материал"] = df["Дней с последней активности"] > 365
        
        except Exception as e:
            st.error(f"Ошибка при добавлении признаков: {str(e)}")
        
        return df