import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    """
    Класс для предварительной обработки данных
    """
    
    def __init__(self, role_names=None):
        """role_names: dict с ключами 'ROLE_ID','ROLE_DATE','ROLE_TARGET','ROLE_QTY','ROLE_CURRENCY','ROLE_RATE'"""
        self.role_names = role_names or {
            'ROLE_ID': 'ID',
            'ROLE_DATE': 'Дата',
            'ROLE_TARGET': 'Целевая Колонка',
            'ROLE_QTY': 'Количество',
            'ROLE_CURRENCY': 'Валюта',
            'ROLE_RATE': 'Курс'
        }
    
    def process_data(self, data, column_mapping=None):
        """
        Выполняет предварительную обработку данных
        
        Args:
            data: pandas DataFrame с исходными данными
            column_mapping: dict или None. Ожидается словарь с ключами:
                'ID','Дата','Целевая','Количество','ЕЦЗ','Валюта','Курс'
                значения — имена колонок в загруженных данных или None.
                Метод переименует выбранные колонки в внутренние имена:
                ID -> 'Материал', Дата -> 'ДатаСоздан', Целевая -> 'Цена нетто',
                Количество -> 'за', ЕЦЗ -> 'ЕЦЗ', Валюта -> 'Влт', Курс -> 'Курс'.
        
        Returns:
            pandas DataFrame с обработанными данными
        """
        # Создаем копию данных, чтобы не изменять исходный DataFrame
        df = data.copy()

        # Обязуем использовать только переданный маппинг: все дальнейшие операции
        # работают с роль-именами: 'ID','Дата','Целевая','Количество','ЕЦЗ','Валюта','Курс'.
        # Если маппинг не передан — прекращаем обработку.
        # Роли, используемые в процессе обработки
        ROLE_ID = self.role_names.get('ROLE_ID')
        ROLE_DATE = self.role_names.get('ROLE_DATE')
        ROLE_TARGET = self.role_names.get('ROLE_TARGET')
        ROLE_QTY = self.role_names.get('ROLE_QTY')
        ROLE_CURRENCY = self.role_names.get('ROLE_CURRENCY')
        ROLE_RATE = self.role_names.get('ROLE_RATE')
        roles = [ROLE_ID, ROLE_DATE, ROLE_TARGET, ROLE_QTY, ROLE_CURRENCY, ROLE_RATE]
        if not column_mapping:
            st.error("Не задан маппинг колонок. Обработка требует указания соответствия ролей колонкам.")
            return df

        try:
            # Для корректного переименования требуется, чтобы один источник не назначался
            # на несколько ролей (т.к. одноимённое переименование в несколько имён невозможно).
            selected_sources = [v for v in column_mapping.values() if v is not None]
            if len(selected_sources) != len(set(selected_sources)):
                st.error("В маппинге есть дубли: один источник назначен на несколько ролей. Переименование невозможно — назначьте уникальные источники для каждой роли.")
                return df

            # Переименовываем исходные колонки в имена ролей на месте (без создания новых столбцов)
            renamed = []
            for role in roles:
                src = column_mapping.get(role)
                if src is None:
                    continue
                if src not in df.columns:
                    st.error(f"Выбранная колонка '{src}' для роли '{role}' не найдена в загруженных данных.")
                    return df
                if src == role:
                    # уже имеет нужное имя
                    continue
                # если целевое имя роли уже занято другим столбцом (и это не тот же источник) — конфликт
                if role in df.columns and role != src:
                    st.error(f"Невозможно переименовать '{src}' в '{role}': колонка с именем роли уже существует и не совпадает с источником.")
                    return df
                # выполняем переименование
                df.rename(columns={src: role}, inplace=True)
                renamed.append((src, role))

            st.info(f"Маппинг применён: {', '.join([f'{r}: {column_mapping.get(r)}' for r in roles if column_mapping.get(r)])}")
            if renamed:
                st.info(f"Переименованы колонки: {', '.join([f'{s} -> {t}' for s,t in renamed])}")
        except Exception as e:
            st.warning(f"Ошибка при применении маппинга колонок: {e}")

        # Обработка колонок по ролям
        df = self._process_id_column(df, ROLE_ID)
        df = self._process_date_column(df, ROLE_DATE)
        df = self._process_target_column(df, ROLE_TARGET)
        df = self._process_exchange_rate_column(df, ROLE_RATE)

        # Нормализация цен в единую валюту (если несколько валют)
        df = self._normalize_prices(df, ROLE_TARGET, ROLE_RATE, ROLE_CURRENCY, ROLE_QTY, column_mapping=column_mapping)

        # Заполнение пропущенных значений
        df = self._fill_missing_values(df, ROLE_ID)

        # Добавление дополнительных признаков
        df = self._add_features(df, ROLE_ID, ROLE_DATE, ROLE_TARGET)

        return df
    
    def _process_id_column(self, df, role_id):
        """Обрабатывает идентификационную колонку (role_id)"""
        if role_id not in df.columns:
            st.error(f"Идентификационная колонка '{role_id}' не найдена в данных")
            return df

        df[role_id] = df[role_id].astype(str).str.strip()
        return df
    
    def _process_date_column(self, df, role_date):
        """Обрабатывает колонку с датой (role_date) и приводит к datetime"""
        if role_date not in df.columns:
            st.error(f"Колонка даты '{role_date}' не найдена в данных")
            return df

        try:
            date_formats = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]
            for date_format in date_formats:
                try:
                    df[role_date] = pd.to_datetime(df[role_date], format=date_format)
                    break
                except Exception:
                    continue

            if not pd.api.types.is_datetime64_dtype(df[role_date]):
                df[role_date] = pd.to_datetime(df[role_date], errors="coerce")

            if df[role_date].isna().any():
                st.warning(f"В колонке '{role_date}' обнаружены некорректные значения. Количество пропущенных: {df[role_date].isna().sum()}")

        except Exception as e:
            st.error(f"Ошибка при обработке колонки даты '{role_date}': {str(e)}")

        return df
    
    def _process_target_column(self, df, role_target):
        """Обрабатывает целевую колонку (role_target) — приведение к числу"""
        if role_target not in df.columns:
            st.error(f"Целевая колонка '{role_target}' не найдена в данных")
            return df

        try:
            df[role_target] = df[role_target].astype(str)
            df[role_target] = df[role_target].str.replace(" ", "").str.replace(",", ".")
            df[role_target] = pd.to_numeric(df[role_target], errors="coerce")

            if (df[role_target] < 0).any():
                st.warning(f"В целевой колонке '{role_target}' обнаружены отрицательные значения. Количество: {(df[role_target] < 0).sum()}")
            if df[role_target].isna().any():
                st.warning(f"В целевой колонке '{role_target}' обнаружены пропущенные значения. Количество: {df[role_target].isna().sum()}")

        except Exception as e:
            st.error(f"Ошибка при обработке целевой колонки '{role_target}': {str(e)}")

        return df
    
    def _process_exchange_rate_column(self, df, role_rate):
        """Обрабатывает колонку курса валюты (role_rate)"""
        # Если колонка курса не указана, создаём колонку с единицами
        if role_rate not in df.columns:
            st.info(f"Колонка курса '{role_rate}' не указана — все значения будут считаться как 1.0")
            df[role_rate] = 1.0
            return df

        try:
            df[role_rate] = df[role_rate].astype(str).str.replace(" ", "").str.replace(",", ".")
            df[role_rate] = pd.to_numeric(df[role_rate], errors="coerce")
            df[role_rate] = df[role_rate].fillna(1.0)
            if (df[role_rate] <= 0).any():
                st.warning(f"В колонке курса '{role_rate}' обнаружены нулевые или отрицательные значения — заменяю на 1.")
                df.loc[df[role_rate] <= 0, role_rate] = 1.0

        except Exception as e:
            st.error(f"Ошибка при обработке колонки курса '{role_rate}': {str(e)}")

        return df
    
    def _normalize_prices(self, df, role_target, role_rate, role_currency, role_qty, column_mapping=None):
        """Нормализует целевую цену по курсу и количеству (в роли).
        Возвращает df с новой колонкой f"{role_target} (норм.)" и колонкой базовой валюты.
        """
        # role_target is required; rate, currency, qty are optional
        if role_target not in df.columns:
            st.error(f"Целевая колонка '{role_target}' не найдена в данных для нормализации")
            return df

        try:
            # Обработка колонки курса: используем маппинг — если роль 'Курс' была назначена в mapping,
            # то используем колонку role_rate (она уже переименована в role_rate при применении mapping);
            # если маппинга нет — игнорируем колонку 'Курс' даже если она есть в датасете.
            if column_mapping and column_mapping.get(role_rate) is not None:
                # ожидаем, что ранее мы переименовали колонку-источник в имя роли, поэтому берем df[role_rate]
                rate_series = df[role_rate].astype(str).str.replace(" ", "").str.replace(",", ".")
                rate_series = pd.to_numeric(rate_series, errors="coerce").fillna(1.0)
                # заменим невалидные/<=0 на 1.0
                rate_series.loc[rate_series <= 0] = 1.0
            else:
                rate_series = pd.Series(1.0, index=df.index)

            # Обработка количества: если есть — привести к числу и заменить 0/NaN на 1, иначе единицы
            if column_mapping and column_mapping.get(role_qty) is not None:
                qty_series = df[role_qty].astype(str).str.replace(" ", "").str.replace(",", ".")
                qty_series = pd.to_numeric(qty_series, errors="coerce")
                qty_series = qty_series.replace(0, 1).fillna(1)
            else:
                qty_series = pd.Series(1.0, index=df.index)

            # Убедимся, что целевая колонка числовая (предварительная очистка в _process_target_column должна помочь)
            target_series = df[role_target]
            try:
                target_series = pd.to_numeric(target_series, errors="coerce")
            except Exception:
                # если не удалось привести — оставить как есть и попытаться вычислить (результатом будут NaN там, где нечисло)
                target_series = pd.to_numeric(target_series.astype(str).str.replace(" ", "").str.replace(",", "."), errors="coerce")

            # Нормализованная целевая цена: умножаем на курс (если есть) и делим на количество (если есть)
            norm_col = f"{role_target} (норм.)"
            df[norm_col] = (target_series.abs() * rate_series) / qty_series

            # Базовая валюта: если указана колонка валюты — возьмём наиболее частую, иначе RUB
            if column_mapping and column_mapping.get(role_currency) is not None:
                try:
                    base_currency = df[role_currency].mode().iloc[0]
                    df["Базовая валюта"] = base_currency
                    st.info(f"Базовая валюта для нормализации цен: {base_currency}")
                except Exception:
                    df["Базовая валюта"] = "RUB"
            else:
                df["Базовая валюта"] = "RUB"

        except Exception as e:
            st.error(f"Ошибка при нормализации цен: {str(e)}")

        return df
    
    def _fill_missing_values(self, df, role_id):
        """
        Заполняет пропущенные значения
        """
        # Проверяем, есть ли пропущенные значения
        if df.isna().any().any():
            st.warning("В данных обнаружены пропущенные значения")

            # Заполняем пропущенные значения в числовых колонках медианой по группе идентификатора
            numeric_columns = df.select_dtypes(include=["number"]).columns
            for col in numeric_columns:
                if df[col].isna().any():
                    if role_id in df.columns:
                        df[col] = df.groupby(role_id)[col].transform(
                            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
                        )
                    else:
                        df[col] = df[col].fillna(df[col].median())

            if df.isna().any().any():
                st.warning("Некоторые пропущенные значения остались после заполнения")

        return df
    
    def _add_features(self, df, role_id, role_date, role_target):
        """
        Добавляет дополнительные признаки
        """
        try:
            # Добавляем год, месяц и день на основе role_date
            if role_date in df.columns:
                df["Год"] = df[role_date].dt.year
                df["Месяц"] = df[role_date].dt.month
                df["День"] = df[role_date].dt.day

                df["Квартал"] = df[role_date].dt.quarter
                df["День недели"] = df[role_date].dt.dayofweek

                min_date = df[role_date].min()
                df["Дней от начала"] = (df[role_date] - min_date).dt.days
            
            # Добавляем признак сезонности (зима, весна, лето, осень)
            season_map = {
                1: "Зима", 2: "Зима", 3: "Весна", 4: "Весна", 5: "Весна",
                6: "Лето", 7: "Лето", 8: "Лето", 9: "Осень", 10: "Осень",
                11: "Осень", 12: "Зима"
            }
            df["Сезон"] = df["Месяц"].map(season_map)
            
            # Добавляем количество записей для каждой ID-группы
            if role_id in df.columns:
                df["Количество записей"] = df.groupby(role_id)[role_id].transform("count")

                # Средняя и стд по нормализованной цене (если есть)
                norm_col = f"{role_target} (норм.)"
                if norm_col in df.columns:
                    df["Средняя цена"] = df.groupby(role_id)[norm_col].transform("mean")
                    df["Стд. отклонение цены"] = df.groupby(role_id)[norm_col].transform("std")
                    df["Коэффициент вариации цены"] = (df["Стд. отклонение цены"] / df["Средняя цена"]) * 100
                    df["Коэффициент вариации цены"] = df["Коэффициент вариации цены"].replace([np.inf, -np.inf], np.nan).fillna(0)

                    def is_stable_price(group):
                        most_common_value_count = group.value_counts().iloc[0] if len(group.value_counts()) > 0 else 0
                        return most_common_value_count / len(group) >= 0.8

                    stable_flags = df.groupby(role_id)[norm_col].apply(is_stable_price)
                    df["Стабильная цена"] = df[role_id].map(stable_flags)

                # Временной диапазон и последняя активность
                if role_date in df.columns:
                    df["Временной диапазон"] = df.groupby(role_id)[role_date].transform(lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
                    latest_date = df[role_date].max()
                    df["Последняя активность"] = df.groupby(role_id)[role_date].transform("max")
                    df["Дней с последней активности"] = (latest_date - df["Последняя активность"]).dt.days
                    df["Неактивный"] = df["Дней с последней активности"] > 365
        
        except Exception as e:
            st.error(f"Ошибка при добавлении признаков: {str(e)}")
        
        return df