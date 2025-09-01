import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io  # Add this import for BytesIO

class SecurityAnalyzer:
    """
    Класс для анализа данных на предмет потенциальных мошеннических схем
    и организационных изменений для служб информационной безопасности
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'high_volatility': 50,         # Высокий коэффициент вариации
            'suspicious_stable': 0.5,      # Подозрительно стабильное значение
            'price_jump': 200,             # Резкий скачок значения (%)
            'purchase_frequency': 3,       # Подозрительная частота событий (дни)
            'end_period_activity': 3,      # Активность в конце периода (дни)
            'quarterly_increase': 25,      # Повышение в конце квартала (%)
            'round_price_tolerance': 0.01  # Допуск для округленных значений (%)
        }

        # role_names will be injected from app; default to legacy names if not provided
        self.role_names = {
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

    def set_role_names(self, role_names: dict):
        """Установить ROLE_NAMES из app (позволяет backward-compatibility)

        Это метод вызывается из app при создании SecurityAnalyzer, если требуется.
        """
        if not role_names:
            return
        self.role_names.update(role_names)
        self.ROLE_ID = self.role_names.get('ROLE_ID')
        self.ROLE_DATE = self.role_names.get('ROLE_DATE')
        self.ROLE_TARGET = self.role_names.get('ROLE_TARGET')

    def _resolve_columns(self, df: pd.DataFrame):
        """Resolve identifier, date and price column names in a dataframe.

        Preference order: ROLE_NAMES values -> legacy names -> sensible fallback (first column or None)
        Returns: (id_col, date_col, price_col)
        """
        # identifier
        if self.ROLE_ID in df.columns:
            id_col = self.ROLE_ID
        elif 'Материал' in df.columns:
            id_col = 'Материал'
        else:
            id_col = df.columns[0]

        # date
        if self.ROLE_DATE in df.columns:
            date_col = self.ROLE_DATE
        elif 'ДатаСоздан' in df.columns:
            date_col = 'ДатаСоздан'
        else:
            date_col = None

        # price - prefer normalized column
        norm_name = f"{self.ROLE_TARGET} (норм.)"
        if norm_name in df.columns:
            price_col = norm_name
        elif self.ROLE_TARGET in df.columns:
            price_col = self.ROLE_TARGET
        elif 'Цена нетто (норм.)' in df.columns:
            price_col = 'Цена нетто (норм.)'
        elif 'Цена нетто' in df.columns:
            price_col = 'Цена нетто'
        else:
            price_col = None

        return id_col, date_col, price_col

    def _apply_reverse_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Попытка вернуть названия колонок к исходным, выбранным пользователем в `st.session_state['column_mapping']`.
        Не меняет DataFrame, если mapping отсутствует или какие-то колонки не найдены.
        """
        try:
            if 'column_mapping' in st.session_state and st.session_state.get('column_mapping'):
                mapping = st.session_state['column_mapping'] or {}
                rename_map = {}
                for canonical_role, original_col in mapping.items():
                    if not original_col:
                        continue
                    # если каноническое имя колонкой присутствует — переименуем
                    if canonical_role in df.columns:
                        rename_map[canonical_role] = original_col
                    # normalized price column
                    canonical_norm = f"{canonical_role} (норм.)"
                    if canonical_norm in df.columns:
                        rename_map[canonical_norm] = f"{original_col} (норм.)"

                if rename_map:
                    df = df.rename(columns=rename_map)
        except Exception:
            # в случае ошибки возвращаем исходный df
            return df

        return df
    
    def analyze_security_risks(self, data, segments):
        """
        Проводит комплексный анализ безопасности данных - оптимизированная версия
        
        Args:
            data: pandas DataFrame с обработанными данными
            segments: словарь с сегментами временных рядов
        
        Returns:
            DataFrame с оценкой рисков для временных рядов
        """
        st.header("Анализ данных для служб информационной безопасности")

        st.markdown("""
        Этот раздел предназначен для выявления потенциальных мошеннических схем и аномалий в данных.
        Система анализирует различные паттерны и индикаторы, которые могут свидетельствовать о нарушениях
        или требовать дополнительной проверки.
        """)

        # Прогресс-бар и статус
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Начинаем анализ безопасности...")

        try:
            # Используем Series вместо массива для unique, чтобы не создавать лишний объект
            id_col, date_col, price_col = self._resolve_columns(data)
            unique_materials = data[id_col].unique()
            total_materials = len(unique_materials)

            # Для очень больших объемов данных предлагаем оптимизированный анализ
            if (total_materials > 10000):
                limit_analysis = st.checkbox(
                    f"Обнаружено очень много временных рядов ({total_materials}). Ограничить анализ до 5000 случайных рядов?",
                    value=True
                )
                if limit_analysis:
                    # Используем numpy.random.Generator вместо numpy.random для лучшей производительности
                    rng = np.random.default_rng(42)  # Современная замена np.random.seed
                    unique_materials = rng.choice(unique_materials, size=5000, replace=False)
                    total_materials = 5000
                    st.info(f"Анализ ограничен до {total_materials} случайных временных рядов")

            # Используем пустой список для накопления результатов
            risk_data = []

            # Оптимизируем размер батча в зависимости от объема данных
            batch_size = min(1000, max(100, total_materials // 10))  # Увеличиваем размер для меньшего числа итераций

            # Создаем словарь для быстрого поиска сегмента - используем dict comprehension для оптимизации
            material_to_segment = {}
            for segment_name, segment_data in segments.items():
                seg_id_col, _, _ = self._resolve_columns(segment_data)
                material_to_segment.update({m: segment_name for m in segment_data[seg_id_col].values})

            # Мониторим память только если необходимо
            memory_warning_shown = False
            try:
                import psutil
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                process = None

            # Обработка по оптимизированным чанкам
            for i in range(0, total_materials, batch_size):
                batch_end = min(i + batch_size, total_materials)
                batch_materials = unique_materials[i:batch_end]

                # Обновляем прогресс реже для ускорения обработки
                if i % (batch_size * 2) == 0 or i == 0:
                    progress_value = i / total_materials
                    progress_bar.progress(progress_value)
                    status_text.text(f"Анализ временных рядов... {i}/{total_materials}")

                # Оптимизируем фильтрацию данных для батча
                batch_data = data[data[id_col].isin(batch_materials)]

                # Предварительная группировка по ID для ускорения
                grouped_data = dict(list(batch_data.groupby(id_col)))

                # Обрабатываем каждый временной ряд
                for material in batch_materials:
                    if material not in grouped_data:
                        continue

                    material_data = grouped_data[material]
                    if date_col is not None and date_col in material_data.columns:
                        material_data = material_data.sort_values(date_col)

                    if len(material_data) < 2:
                        continue

                    # Вычисляем метрики риска
                    risk_metrics = self._calculate_risk_metrics(material_data, price_col=price_col, date_col=date_col)

                    # Определяем категорию риска
                    risk_category, risk_factors = self._determine_risk_category(risk_metrics)

                    # Находим сегмент
                    material_segment = material_to_segment.get(material, "Не определен")

                    # Сохраняем только необходимые данные (ключ идентификатора используем ROLE_ID)
                    risk_data.append({
                        id_col: material,
                        'Категория риска': risk_category,
                        'Факторы риска': risk_factors,
                        'Сегмент': material_segment,
                        'Количество записей': len(material_data),
                        'Коэффициент вариации': risk_metrics['volatility'],
                        'Индекс аномальности значения': risk_metrics['price_anomaly_index'],
                        'Индекс дробления событий': risk_metrics['purchase_fragmentation'],
                        'Индекс сезонных отклонений': risk_metrics.get('seasonal_deviation', 0),  # Используем get для безопасного доступа
                        'Индекс подозрительности': risk_metrics['suspicion_index']
                    })

                # Проверяем использование памяти только каждые 5 батчей
                if process and i % (batch_size * 5) == 0 and not memory_warning_shown:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_diff = current_memory - initial_memory
                    if memory_diff > 500:
                        st.warning(f"Высокое потребление памяти: {memory_diff:.1f} МБ. Если анализ слишком медленный, попробуйте ограничить выборку.")
                        memory_warning_shown = True  # Показываем предупреждение только один раз

            # Создаем DataFrame из собранных данных один раз
            risk_df = pd.DataFrame(risk_data) if risk_data else pd.DataFrame()

            # Сортируем только если есть данные
            if not risk_df.empty:
                risk_df = risk_df.sort_values('Индекс подозрительности', ascending=False)

            progress_bar.progress(1.0)
            status_text.text("Анализ завершен!")

            if risk_df.empty:
                st.warning("Не удалось обнаружить временные ряды с признаками риска")
                return pd.DataFrame()

            # Отображаем результаты
            # self._display_security_analysis_results(risk_df) # УДАЛЕНО

            return risk_df

        except Exception as e:
            from modules.utils import show_error_message
            show_error_message(e, "Ошибка при анализе безопасности", show_traceback=True)
            st.info("Попробуйте ограничить количество анализируемых временных рядов или использовать меньший набор данных.")
            return pd.DataFrame()
    
    def _calculate_risk_metrics(self, material_data, price_col=None, date_col=None):
        """
        Вычисляет метрики риска для временного ряда - исправленная и role-aware версия

        Args:
            material_data: DataFrame с данными по одному временному ряду
            price_col: имя целевой колонки (опционально)
            date_col: имя колонки с датой (опционально)

        Returns:
            dict: словарь с метриками риска
        """
        metrics = {}

        # Resolve columns if not provided
        if price_col is None or date_col is None:
            resolved_id, resolved_date, resolved_price = self._resolve_columns(material_data)
            if price_col is None:
                price_col = resolved_price
            if date_col is None:
                date_col = resolved_date

        # Prices as numpy array
        prices = material_data[price_col].values if (price_col and price_col in material_data.columns) else np.array([])

        # Basic stats
        if prices.size > 0:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            min_price = np.min(prices)
            max_price = np.max(prices)
            metrics['volatility'] = (std_price / mean_price) * 100 if mean_price > 0 else 0
            metrics['price_anomaly_index'] = (max_price / min_price) if min_price > 0 else 1
        else:
            metrics['volatility'] = 0
            metrics['price_anomaly_index'] = 1

        # Event frequency / fragmentation
        dates = material_data[date_col].values if (date_col and date_col in material_data.columns) else np.array([])
        if dates.size > 1:
            if not np.all(np.diff(dates) >= np.timedelta64(0)):
                dates = np.sort(dates)
            date_diffs = np.diff(dates) / np.timedelta64(1, 'D')
            metrics['avg_days_between_purchases'] = np.mean(date_diffs) if date_diffs.size > 0 else 0
            small_intervals = np.sum(date_diffs <= self.risk_thresholds['purchase_frequency'])
            metrics['purchase_fragmentation'] = small_intervals / date_diffs.size * 100 if date_diffs.size > 0 else 0
        else:
            metrics['avg_days_between_purchases'] = 0
            metrics['purchase_fragmentation'] = 0

        # Seasonality
        if len(material_data) >= 12 and (date_col and price_col) and prices.size > 0:
            months = material_data[date_col].dt.month.values
            unique_months = np.unique(months)
            if len(unique_months) > 1:
                monthly_means = np.array([np.mean(prices[months == month]) for month in unique_months])
                mean_of_means = np.mean(monthly_means)
                std_of_means = np.std(monthly_means)
                metrics['seasonal_deviation'] = (std_of_means / mean_of_means) * 100 if mean_of_means > 0 else 0
            else:
                metrics['seasonal_deviation'] = 0
        else:
            metrics['seasonal_deviation'] = 0

        # End-of-period activity
        if date_col and date_col in material_data.columns:
            month_values = material_data[date_col].dt.month.values
            day_values = material_data[date_col].dt.day.values
            end_of_month = np.mean(day_values >= 28) * 100
            end_of_quarter_months = np.array([3, 6, 9, 12])
            is_quarter_end = np.isin(month_values, end_of_quarter_months)
            is_month_end = day_values >= 28
            end_of_quarter = np.mean(is_quarter_end & is_month_end) * 100 if len(is_quarter_end) > 0 else 0
        else:
            end_of_month = 0
            end_of_quarter = 0

        metrics['end_of_month_activity'] = end_of_month
        metrics['end_of_quarter_activity'] = end_of_quarter

        # Rounded value detection
        unique_prices = np.unique(prices) if prices.size > 0 else np.array([])
        rounded_prices_count = 0
        if unique_prices.size > 0:
            if len(unique_prices) < 50:
                for price in unique_prices:
                    if price <= 0:
                        continue
                    for magnitude in [10, 100, 1000, 10000, 100000]:
                        if abs(price % magnitude) / price < self.risk_thresholds['round_price_tolerance']:
                            rounded_prices_count += 1
                            break
            else:
                processed = np.zeros(len(unique_prices), dtype=bool)
                for magnitude in [10, 100, 1000, 10000, 100000]:
                    valid_prices = (unique_prices > 0) & (~processed)
                    if not np.any(valid_prices):
                        break
                    valid_indices = np.where(valid_prices)[0]
                    valid_price_values = unique_prices[valid_indices]
                    remainder_ratio = np.abs(valid_price_values % magnitude) / valid_price_values
                    rounded_mask = remainder_ratio < self.risk_thresholds['round_price_tolerance']
                    rounded_indices = valid_indices[rounded_mask]
                    rounded_prices_count += len(rounded_indices)
                    processed[rounded_indices] = True

            unique_price_count = len(unique_prices)
            metrics['rounded_prices_ratio'] = rounded_prices_count / unique_price_count * 100 if unique_price_count > 0 else 0
        else:
            metrics['rounded_prices_ratio'] = 0

        # Suspicion index
        suspicion_index = (
            min(metrics['volatility'] * 0.5, 50) +
            min((metrics['price_anomaly_index'] - 1) * 10, 30) +
            min(metrics['purchase_fragmentation'] * 0.5, 30) +
            min(metrics.get('end_of_quarter_activity', 0) * 0.5, 20) +
            min(metrics.get('rounded_prices_ratio', 0) * 0.3, 20)
        )
        metrics['suspicion_index'] = min(suspicion_index, 100)

        return metrics
    
    def _determine_risk_category(self, risk_metrics):
        """
        Определяет категорию риска на основе метрик
        
        Args:
            risk_metrics: словарь с метриками риска
            
        Returns:
            tuple: (категория риска, список факторов риска)
        """
        # Инициализируем список факторов риска
        risk_factors = []
        
        # Проверяем различные факторы риска
        if risk_metrics['volatility'] > self.risk_thresholds['high_volatility']:
            risk_factors.append("Высокая волатильность значений")
        
        if risk_metrics['price_anomaly_index'] > 3:
            risk_factors.append("Значительные скачки значений")
        
        if risk_metrics['purchase_fragmentation'] > 30:
            risk_factors.append("Признаки дробления событий")
        
        if risk_metrics['end_of_quarter_activity'] > 40:
            risk_factors.append("Повышенная активность в конце кварталов")
        
        if risk_metrics['rounded_prices_ratio'] > 70:
            risk_factors.append("Подозрительно округленные значения")
        
        # Определяем категорию риска
        if risk_metrics['suspicion_index'] >= 70:
            risk_category = "Высокий"
        elif risk_metrics['suspicion_index'] >= 40:
            risk_category = "Средний"
        else:
            risk_category = "Низкий"
        
        return risk_category, ", ".join(risk_factors) if risk_factors else "Нет выявленных факторов"
    
    def display_security_analysis_results(self, risk_df):
        """
        Отображает результаты анализа безопасности
        
        Args:
            risk_df: DataFrame с данными о рисках
        """
        # Resolve identifier column and a friendly display name
        if risk_df is None or risk_df.empty:
            st.warning("Результаты анализа безопасности отсутствуют или пусты.")
            return

        id_col, _, _ = self._resolve_columns(risk_df)
        display_id_label = st.session_state.get('column_mapping', {}).get('ID', id_col) if 'column_mapping' in st.session_state else id_col

        # 1. Общая статистика
        st.subheader("Общая статистика анализа безопасности")

        total_items = len(risk_df)
        high_risk = (risk_df['Категория риска'] == 'Высокий').sum()
        medium_risk = (risk_df['Категория риска'] == 'Средний').sum()
        low_risk = (risk_df['Категория риска'] == 'Низкий').sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Временные ряды с высоким риском", f"{high_risk} ({high_risk/total_items*100:.1f}%)")
        with col2:
            st.metric("Временные ряды со средним риском", f"{medium_risk} ({medium_risk/total_items*100:.1f}%)")
        with col3:
            st.metric("Временные ряды с низким риском", f"{low_risk} ({low_risk/total_items*100:.1f}%)")

        # 2. Распределение рисков по сегментам
        st.subheader("Распределение рисков по сегментам")
        risk_by_segment = pd.crosstab(risk_df['Сегмент'], risk_df['Категория риска'], normalize='index') * 100
        risk_by_segment_plot = risk_by_segment.reset_index().melt(id_vars=['Сегмент'], var_name='Категория риска', value_name='Процент временных рядов')

        fig = px.bar(
            risk_by_segment_plot,
            x='Сегмент',
            y='Процент временных рядов',
            color='Категория риска',
            title='Распределение категорий риска по сегментам',
            color_discrete_map={'Высокий': '#FF4B4B', 'Средний': '#FFA72B', 'Низкий': '#2ECC71'}
        )
        fig.update_layout(xaxis_title='Сегмент', yaxis_title='Процент временных рядов (%)', barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

        # 3. Топ временных рядов с высоким риском
        st.subheader("Топ временных рядов с высоким индексом подозрительности")
        high_risk_materials = risk_df.sort_values('Индекс подозрительности', ascending=False).head(20)
        if high_risk_materials.empty:
            st.warning("Нет данных для отображения топа временных рядов с риском.")
        else:
            for _, row in high_risk_materials.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([3,2,5])
                    with c1:
                        try:
                            identifier_value = row[id_col]
                        except Exception:
                            identifier_value = row.get(self.ROLE_ID, '')
                        st.write(f"**{display_id_label}:** {identifier_value}")
                        st.write(f"**Сегмент:** {row['Сегмент']}")
                    with c2:
                        risk_color = "#FF4B4B" if row['Категория риска'] == "Высокий" else "#FFA72B" if row['Категория риска'] == "Средний" else "#2ECC71"
                        st.markdown(f"""
                        <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; color: white;">
                            <strong>Риск:</strong> {row['Категория риска']}<br>
                            <strong>Индекс:</strong> {row['Индекс подозрительности']:.1f}
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"**Факторы риска:** {row['Факторы риска']}")
                    st.divider()

        # 4. Фильтр и таблица
        st.subheader("Поиск временных рядов по уровню риска")
        col1, col2 = st.columns(2)
        with col1:
            risk_category_filter = st.selectbox("Категория риска:", ["Все", "Высокий", "Средний", "Низкий"], key='security_risk_category_filter')
        with col2:
            segment_filter = st.selectbox("Сегмент:", ["Все"] + list(risk_df['Сегмент'].unique()), key='security_segment_filter')

        filtered_risk_df = risk_df.copy()
        if risk_category_filter != "Все":
            filtered_risk_df = filtered_risk_df[filtered_risk_df['Категория риска'] == risk_category_filter]
        if segment_filter != "Все":
            filtered_risk_df = filtered_risk_df[filtered_risk_df['Сегмент'] == segment_filter]

        from modules.utils import create_styled_dataframe
        st.dataframe(create_styled_dataframe(filtered_risk_df, highlight_cols=['Индекс подозрительности'], highlight_threshold=70, precision=2), use_container_width=True, height=500)

        # 5. Экспорт данных
        if not filtered_risk_df.empty:
            # Пытаемся вернуть исходные имена колонок перед экспортом
            export_df = self._apply_reverse_column_mapping(filtered_risk_df.copy())
            buffer = io.BytesIO()
            export_df.to_csv(buffer, index=False, encoding='utf-8-sig')
            buffer.seek(0)
            st.download_button(label="Скачать отчет о рисках (CSV)", data=buffer, file_name="security_risk_report.csv", mime="text/csv; charset=utf-8-sig", key="security_csv_main")

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                # Переименовываем колонки обратно для экспорта в Excel
                export_df = self._apply_reverse_column_mapping(filtered_risk_df.copy())
                export_df.to_excel(writer, sheet_name='Отчет о рисках', index=False)
                workbook = writer.book
                worksheet = writer.sheets['Отчет о рисках']
                header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'bg_color': '#D7E4BC', 'border': 1})
                for col_num, value in enumerate(filtered_risk_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                for i, col in enumerate(filtered_risk_df.columns):
                    column_width = max(filtered_risk_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, column_width)
            excel_buffer.seek(0)
            st.download_button(label="Скачать отчет о рисков (Excel)", data=excel_buffer, file_name="security_risk_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="security_excel_main")
        if not filtered_risk_df.empty:
            # повторный блок экспорта (UI duplication) — тоже применяем reverse mapping
            export_df = self._apply_reverse_column_mapping(filtered_risk_df.copy())
            buffer = io.BytesIO()
            export_df.to_csv(buffer, index=False, encoding='utf-8-sig')
            buffer.seek(0)
            st.download_button(
                label="Скачать отчет о рисках (CSV)",
                data=buffer,
                file_name="security_risk_report.csv",
                mime="text/csv; charset=utf-8-sig",
                key="security_csv_dup"
            )
            
            # Добавляем кнопку для скачивания Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                export_df = self._apply_reverse_column_mapping(filtered_risk_df.copy())
                export_df.to_excel(writer, sheet_name='Отчет о рисках', index=False)
                
                # Базовое форматирование Excel
                workbook = writer.book
                worksheet = writer.sheets['Отчет о рисках']
                
                # Формат для заголовков
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Применяем формат заголовка
                for col_num, value in enumerate(filtered_risk_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Автоподбор ширины столбцов
                for i, col in enumerate(filtered_risk_df.columns):
                    column_width = max(
                        filtered_risk_df[col].astype(str).map(len).max(), 
                        len(col)
                    ) + 2
                    worksheet.set_column(i, i, column_width)
            
            excel_buffer.seek(0)
            st.download_button(
                label="Скачать отчет о рисков (Excel)",
                data=excel_buffer,
                file_name="security_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="security_excel_dup"
            )
    
    def highlight_suspicious_materials(self, data, material_code=None):
        """
        Визуализирует подозрительные паттерны для конкретного временного ряда
        
        Args:
            data: pandas DataFrame с обработанными данными
            material_code: ID временного ряда для анализа (если None, будет предложен выбор)
        """
        st.subheader("Детальный анализ подозрительных паттернов")
        
        # Если ID не указан, предлагаем выбор
        id_col, date_col, price_col = self._resolve_columns(data)
        if material_code is None:
            # Получаем список временных рядов с наибольшей волатильностью
            volatile_materials = data.groupby(id_col)[price_col].agg(['mean', 'std']).reset_index()
            volatile_materials['volatility'] = volatile_materials['std'] / volatile_materials['mean'] * 100
            volatile_materials = volatile_materials.sort_values('volatility', ascending=False).head(50)
            
            material_code = st.selectbox(
                "Выберите временной ряд для анализа:",
                volatile_materials[id_col].tolist()
            )
        
        # Фильтруем данные по выбранному ID
        material_data = data[data[id_col] == material_code].sort_values(date_col)
        
        if material_data.empty:
            st.warning(f"Данные для временного ряда {material_code} не найдены")
            return
        
        # Отображаем основную информацию
        st.write(f"**ID Временного ряда:** {material_code}")
        st.write(f"**Количество записей:** {len(material_data)}")
        st.write(f"**Период данных:** {material_data[date_col].min().strftime('%d.%m.%Y')} - {material_data[date_col].max().strftime('%d.%m.%Y')}")
        
        # Анализируем данные для выявления подозрительных паттернов
        risk_metrics = self._calculate_risk_metrics(material_data)
        risk_category, risk_factors = self._determine_risk_category(risk_metrics)
        
        # Отображаем оценку риска
        risk_color = "#FF4B4B" if risk_category == "Высокий" else "#FFA72B" if risk_category == "Средний" else "#2ECC71"
        st.markdown(f"""
        <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 20px;">
            <h3 style="margin: 0;">Категория риска: {risk_category}</h3>
            <p style="margin: 5px 0 0 0;">Индекс подозрительности: {risk_metrics['suspicion_index']:.1f}</p>
            <p style="margin: 5px 0 0 0;">Факторы риска: {risk_factors}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Визуализация 1: График изменения целевого значения со временем с подсветкой аномалий
        st.subheader("Динамика целевого значения с подсветкой аномалий")
        
        # Вычисляем скользящее среднее и стандартное отклонение
        material_data['rolling_mean'] = material_data[price_col].rolling(window=5, min_periods=1).mean()
        material_data['rolling_std'] = material_data[price_col].rolling(window=5, min_periods=1).std()
        
        # Определяем верхнюю и нижнюю границы для аномалий (2 стандартных отклонения)
        material_data['upper_bound'] = material_data['rolling_mean'] + 2 * material_data['rolling_std']
        material_data['lower_bound'] = material_data['rolling_mean'] - 2 * material_data['rolling_std']
        
        # Помечаем аномалии
        material_data['is_anomaly'] = (material_data[price_col] > material_data['upper_bound']) | \
                                       (material_data[price_col] < material_data['lower_bound'])
        
        # Создаем график
        fig = go.Figure()
        
        # Добавляем линию целевого значения
        fig.add_trace(go.Scatter(
            x=material_data[date_col],
            y=material_data[price_col],
            mode='lines+markers',
            name='Значение',
            line=dict(color='#1976D2', width=2)
        ))
        
        # Добавляем скользящее среднее
        fig.add_trace(go.Scatter(
            x=material_data[date_col],
            y=material_data['rolling_mean'],
            mode='lines',
            name='Скользящее среднее',
            line=dict(color='#2ECC71', width=1, dash='dash')
        ))
        
        # Добавляем границы аномалий
        fig.add_trace(go.Scatter(
            x=material_data[date_col],
            y=material_data['upper_bound'],
            mode='lines',
            name='Верхняя граница',
            line=dict(color='#FF7043', width=1, dash='dot'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=material_data[date_col],
            y=material_data['lower_bound'],
            mode='lines',
            name='Нижняя граница',
            line=dict(color='#FF7043', width=1, dash='dot'),
            showlegend=False,
            fill='tonexty',
            fillcolor='rgba(255, 112, 67, 0.1)'
        ))
        
        # Добавляем аномалии
        if material_data['is_anomaly'].any():
            anomalies = material_data[material_data['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=anomalies[date_col],
                y=anomalies[price_col],
                mode='markers',
                name='Аномалии',
                marker=dict(
                    color='#FF4B4B',
                    size=10,
                    symbol='circle',
                    line=dict(color='#FFFFFF', width=1)
                )
            ))
        
        # Настройка внешнего вида
        fig.update_layout(
            title=f'Динамика целевого значения для {material_code} с подсветкой аномалий',
            xaxis_title='Дата',
            yaxis_title='Целевое значение (норм.)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 2: Анализ периодичности событий
        st.subheader("Анализ периодичности событий")
        
        # Вычисляем разницу между последовательными датами
        material_data['days_diff'] = material_data[date_col].diff().dt.days
        
        # Создаем гистограмму интервалов между событиями
        fig = px.histogram(
            material_data.dropna(),
            x='days_diff',
            nbins=30,
            title='Распределение интервалов между событиями (дни)',
            labels={'days_diff': 'Интервал (дни)', 'count': 'Количество случаев'},
            color_discrete_sequence=['#1976D2']
        )
        
        # Добавляем вертикальную линию для порогового значения дробления
        fig.add_vline(
            x=self.risk_thresholds['purchase_frequency'],
            line_dash="dash",
            line_color="#FF4B4B",
            annotation_text="Порог дробления событий",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 3: Активность по месяцам и кварталам
        st.subheader("Активность по месяцам и кварталам")
        
        # Добавляем информацию о месяце, квартале и годе
        material_data['Год'] = material_data[date_col].dt.year
        material_data['Месяц'] = material_data[date_col].dt.month
        material_data['Квартал'] = material_data[date_col].dt.quarter
        
        # Создаем сводную таблицу по годам и месяцам
        monthly_activity = material_data.groupby(['Год', 'Месяц']).size().reset_index(name='Количество')
        
        # Преобразуем для тепловой карты
        monthly_activity_pivot = monthly_activity.pivot_table(
            values='Количество',
            index='Год',
            columns='Месяц',
            fill_value=0
        )
        
        # Названия месяцев
        month_names = {
            1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь",
            7: "Июль", 8: "Август", 9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
        }
        
        # Создаем тепловую карту
        fig = go.Figure(data=go.Heatmap(
            z=monthly_activity_pivot.values,
            x=[month_names[m] for m in monthly_activity_pivot.columns],
            y=monthly_activity_pivot.index,
            colorscale='YlOrRd',
            colorbar=dict(title="Количество событий")
        ))
        
        fig.update_layout(
            title='Тепловая карта активности по месяцам',
            xaxis_title='Месяц',
            yaxis_title='Год',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 4: Активность по дням месяца
        material_data['День'] = material_data[date_col].dt.day
        monthly_day_activity = material_data.groupby(['День']).size().reset_index(name='Количество')
        
        fig = px.bar(
            monthly_day_activity,
            x='День',
            y='Количество',
            title='Распределение событий по дням месяца',
            labels={'День': 'День месяца', 'Количество': 'Количество событий'},
            color='Количество',
            color_continuous_scale='YlOrRd'
        )
        
        # Подсвечиваем конец месяца
        fig.add_vrect(
            x0=27.5, 
            x1=31.5,
            fillcolor="#FF4B4B", 
            opacity=0.15,
            layer="below", 
            line_width=0,
            annotation_text="Конец месяца",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 5: Распределение значений (проверка на округленность)
        st.subheader("Анализ распределения значений")
        
        # Создаем гистограмму
        fig = px.histogram(
            material_data,
            x=price_col,
            nbins=30,
            title='Распределение значений',
            labels={price_col: 'Значение', 'count': 'Количество случаев'},
            color_discrete_sequence=['#1976D2']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Проверка на округленные значения
        unique_prices = material_data[price_col].unique()
        rounded_prices = []
        
        for price in unique_prices:
            for magnitude in [10, 100, 1000, 10000, 100000]:
                if abs(price % magnitude) / price < self.risk_thresholds['round_price_tolerance']:
                    rounded_prices.append({
                        'Значение': price,
                        'Округлено до': magnitude
                    })
                    break
        
        if rounded_prices:
            st.subheader("Обнаруженные округленные значения")
            st.write(f"**Процент округленных значений:** {len(rounded_prices) / len(unique_prices) * 100:.1f}%")
            from modules.utils import format_streamlit_dataframe
            st.dataframe(
                format_streamlit_dataframe(pd.DataFrame(rounded_prices)),
                use_container_width=True,
                height=300  # Фиксированная высота для лучшего отображения
            )
            
        # Добавляем кнопку для экспорта детального анализа в Excel
        # --- REMOVED DOWNLOAD BUTTON FROM HERE ---
        # excel_data = self.export_detailed_analysis(data, material_code)
        # if excel_data is not None:
        #     st.download_button(
        #         label="Скачать детальный анализ (Excel)",
        #         data=excel_data,
        #         file_name=f"security_analysis_{material_code}.xlsx",
        #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #     )
    
    # --- Helper methods for Excel Sheet Generation --- 

    def _get_excel_formats(self, workbook):
        """Создает и возвращает стандартные форматы для Excel."""
        formats = {}
        formats['header'] = workbook.add_format({
            'bold': True, 
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D7E4BC',
            'border': 1
        })
        formats['orange_highlight'] = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        formats['red_highlight'] = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        formats['yellow_highlight'] = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'}) # Same as orange for medium risk
        formats['green_highlight'] = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        formats['anomaly_highlight'] = workbook.add_format({'bg_color': '#FF7043', 'font_color': '#FFFFFF'}) # Format for anomaly points
        return formats

    def _auto_adjust_excel_columns(self, df, worksheet, max_width=50):
        """Автоматически подбирает ширину колонок в Excel."""
        for i, col in enumerate(df.columns):
            column_len = max(
                df[col].astype(str).map(len).max(), # Max length of data in column
                len(col) # Length of column header
            ) + 2 # Add a little padding
            column_width = min(column_len, max_width)
            worksheet.set_column(i, i, column_width)

    def _write_general_info_sheet(self, writer, material_code, material_data):
        """Записывает лист 'Общая информация' в Excel."""
        sheet_name = f"{material_code[:20]}_Общая" # Truncate material code if too long
        workbook = writer.book
        formats = self._get_excel_formats(workbook)
        
        _, date_col, price_col = self._resolve_columns(material_data)

        if material_data.empty:
            info_df = pd.DataFrame({'Параметр': ['ID Временного ряда'], 'Значение': [material_code]}) 
            info_df = pd.concat([info_df, pd.DataFrame({'Параметр': ['Ошибка'], 'Значение': ['Нет данных для анализа']})], ignore_index=True)
        else:
            mean_price = material_data[price_col].mean()
            std_price = material_data[price_col].std()
            volatility = (std_price / mean_price * 100) if mean_price > 0 else 0
            info_df = pd.DataFrame({
                'Параметр': [
                    'ID Временного ряда', 'Количество записей', 'Первая дата', 'Последняя дата',
                    'Среднее значение', 'Минимальное значение', 'Максимальное значение',
                    'Стандартное отклонение', 'Коэффициент вариации (%)'
                ],
                'Значение': [
                    material_code,
                    len(material_data),
                    material_data[date_col].min().strftime('%d.%m.%Y'),
                    material_data[date_col].max().strftime('%d.%m.%Y'),
                    f"{mean_price:.2f}",
                    f"{material_data[price_col].min():.2f}",
                    f"{material_data[price_col].max():.2f}",
                    f"{std_price:.2f}",
                    f"{volatility:.2f}%"
                ]
            })

        # Применяем обратный маппинг перед экспортом
        info_df_to_export = self._apply_reverse_column_mapping(info_df.copy())
        info_df_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Apply header format
        for col_num, value in enumerate(info_df_to_export.columns.values):
            worksheet.write(0, col_num, value, formats['header'])

        # Adjust column widths
        worksheet.set_column(0, 0, 30) # Parameter column
        worksheet.set_column(1, 1, 25) # Value column

    def _write_purchase_data_sheet(self, writer, material_code, material_data):
        """Записывает лист 'Данные событий' в Excel."""
        sheet_name = f"{material_code[:20]}_Данные" 
        workbook = writer.book
        formats = self._get_excel_formats(workbook)

        # Prepare data - select relevant columns if necessary
        purchase_df = material_data.copy()

        purchase_df_to_export = self._apply_reverse_column_mapping(purchase_df.copy())
        purchase_df_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Apply header format
        for col_num, value in enumerate(purchase_df_to_export.columns.values):
            worksheet.write(0, col_num, value, formats['header'])

        # Auto-adjust column widths
        self._auto_adjust_excel_columns(purchase_df_to_export, worksheet)

    def _write_anomaly_sheet(self, writer, material_code, material_data):
        """Записывает лист 'Анализ аномалий' в Excel."""
        sheet_name = f"{material_code[:20]}_Аномалии"
        workbook = writer.book
        formats = self._get_excel_formats(workbook)
        
        _, date_col, price_col = self._resolve_columns(material_data)

        if len(material_data) < 2:
            pd.DataFrame({'Сообщение': ['Недостаточно данных для анализа аномалий (менее 2 записей)']}).to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 60)
            return

        # Recalculate anomalies for export
        material_data_copy = material_data.copy()
        material_data_copy['rolling_mean'] = material_data_copy[price_col].rolling(window=5, min_periods=1).mean()
        material_data_copy['rolling_std'] = material_data_copy[price_col].rolling(window=5, min_periods=1).std()
        material_data_copy['upper_bound'] = material_data_copy['rolling_mean'] + 2 * material_data_copy['rolling_std']
        material_data_copy['lower_bound'] = material_data_copy['rolling_mean'] - 2 * material_data_copy['rolling_std']
        material_data_copy['is_anomaly'] = (
            (material_data_copy[price_col] > material_data_copy['upper_bound']) | \
            (material_data_copy[price_col] < material_data_copy['lower_bound'])
        ).fillna(False)
        
        anomalies_export_df = material_data_copy[[
            date_col, price_col, 'rolling_mean', 
            'upper_bound', 'lower_bound', 'is_anomaly'
        ]]
        anomalies_export_df.columns = [
            'Дата', 'Значение', 'Скользящее среднее (5)', 
            'Верхняя граница (2σ)', 'Нижняя граница (2σ)', 'Аномалия'
        ]

        anomalies_export_df_to_export = self._apply_reverse_column_mapping(anomalies_export_df.copy())
        anomalies_export_df_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Apply header format
        for col_num, value in enumerate(anomalies_export_df_to_export.columns.values):
            worksheet.write(0, col_num, value, formats['header'])

        # Highlight anomalies
        for row_num, is_anomaly in enumerate(anomalies_export_df_to_export['Аномалия']):
            if is_anomaly:
                worksheet.conditional_format(row_num + 1, 0, row_num + 1, anomalies_export_df_to_export.shape[1] - 1, 
                                           {'type': 'no_errors', 'format': formats['anomaly_highlight']})

        # Auto-adjust column widths
        self._auto_adjust_excel_columns(anomalies_export_df_to_export, worksheet)

    def _write_periodicity_sheet(self, writer, material_code, material_data):
        """Записывает лист 'Периодичность' в Excel."""
        sheet_name = f"{material_code[:20]}_Периоды"
        workbook = writer.book
        formats = self._get_excel_formats(workbook)
        
        _, date_col, _ = self._resolve_columns(material_data)

        if len(material_data) < 2:
            pd.DataFrame({'Сообщение': ['Недостаточно данных для анализа периодичности (менее 2 записей)']}).to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 60)
            return

        # Recalculate periodicity
        material_data_copy = material_data.copy().sort_values(date_col) # Ensure sorted
        material_data_copy['days_diff'] = material_data_copy[date_col].diff().dt.days
        periods_data = material_data_copy[[date_col, 'days_diff']].dropna()
        periods_data.columns = ['Дата', 'Интервал (дни)']

        periods_data_to_export = self._apply_reverse_column_mapping(periods_data.copy())
        periods_data_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Apply header format
        for col_num, value in enumerate(periods_data_to_export.columns.values):
            worksheet.write(0, col_num, value, formats['header'])

        # Highlight short intervals
        threshold = self.risk_thresholds.get('purchase_frequency', 3) # Use default if not found
        for row_num, interval in enumerate(periods_data_to_export['Интервал (дни)']):
            if interval <= threshold:
                worksheet.write(row_num + 1, 1, interval, formats['orange_highlight'])

        # Auto-adjust column widths
        worksheet.set_column(0, 0, 20) # Date
        worksheet.set_column(1, 1, 15) # Interval

    def _write_risk_assessment_sheet(self, writer, material_code, material_data):
        """Записывает лист 'Оценка рисков' в Excel."""
        sheet_name = f"{material_code[:20]}_Риски"
        workbook = writer.book
        formats = self._get_excel_formats(workbook)

        if len(material_data) < 2:
            risk_data = pd.DataFrame({'Показатель': ['Ошибка'], 'Значение': ['Недостаточно данных для оценки рисков (менее 2 записей)']})
        else:
            risk_metrics = self._calculate_risk_metrics(material_data)
            risk_category, risk_factors = self._determine_risk_category(risk_metrics)
            
            risk_data = pd.DataFrame({
                'Показатель': [
                    'Волатильность значения (%)',
                    'Индекс аномальности значения',
                    'Индекс дробления событий (%)',
                    'Активность в конце месяца (%)',
                    'Активность в конце квартала (%)',
                    'Доля округленных значений (%)',
                    'Общий индекс подозрительности',
                    'Категория риска',
                    'Факторы риска'
                ],
                'Значение': [
                    f"{risk_metrics.get('volatility', 0):.2f}%",
                    f"{risk_metrics.get('price_anomaly_index', 1):.2f}",
                    f"{risk_metrics.get('purchase_fragmentation', 0):.2f}%",
                    f"{risk_metrics.get('end_of_month_activity', 0):.2f}%",
                    f"{risk_metrics.get('end_of_quarter_activity', 0):.2f}%",
                    f"{risk_metrics.get('rounded_prices_ratio', 0):.2f}%",
                    f"{risk_metrics.get('suspicion_index', 0):.2f}",
                    risk_category,
                    risk_factors
                ]
            })

        risk_data_to_export = self._apply_reverse_column_mapping(risk_data.copy())
        risk_data_to_export.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]

        # Apply header format
        for col_num, value in enumerate(risk_data_to_export.columns.values):
            worksheet.write(0, col_num, value, formats['header'])

        # Highlight risk category if calculated
        if len(material_data) >= 2:
            risk_category_row_index = risk_data_to_export[risk_data_to_export['Показатель'] == 'Категория риска'].index[0]
            risk_color_format = None
            if risk_category == "Высокий":
                risk_color_format = formats['red_highlight']
            elif risk_category == "Средний":
                risk_color_format = formats['yellow_highlight']
            else:
                risk_color_format = formats['green_highlight']
            worksheet.write(risk_category_row_index + 1, 1, risk_category, risk_color_format)

        # Auto-adjust column widths
        worksheet.set_column(0, 0, 35) # Indicator column
        worksheet.set_column(1, 1, 60) # Value column (risk factors can be long)

    # --- End of Helper methods --- 

    def export_detailed_analysis(self, data, material_code):
        """
        Экспортирует детальный анализ подозрительного временного ряда в Excel
        (Теперь использует вспомогательные методы)
        
        Args:
            data: pandas DataFrame с обработанными данными
            material_code: ID временного ряда для анализа
            
        Returns:
            bytes: содержимое Excel-файла или None
        """
        # Resolve columns and filter data by canonical identifier
        id_col, date_col, price_col = self._resolve_columns(data)
        if id_col in data.columns:
            material_data = data[data[id_col] == material_code].copy()
        else:
            # fallback to empty
            material_data = pd.DataFrame()

        # Sort by resolved date column if available
        if date_col and date_col in material_data.columns:
            material_data = material_data.sort_values(date_col)
        
        if material_data.empty:
            st.warning(f"Нет данных для временного ряда {material_code} для экспорта.")
            return None
        
        # Before writing, adapt column names expected by helper methods
        def _prepare_for_sheets(df):
            df = df.copy()
            # Map canonical id/date/price columns to legacy names used by helper methods
            if id_col and id_col in df.columns:
                df = df.rename(columns={id_col: 'Материал'}) # Legacy name expected by helpers
            if date_col and date_col in df.columns:
                df = df.rename(columns={date_col: 'ДатаСоздан'}) # Legacy name
            if price_col and price_col in df.columns:
                df = df.rename(columns={price_col: 'Цена нетто (норм.)'}) # Legacy name
            return df

        material_data_for_sheets = _prepare_for_sheets(material_data)

        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                # Use prepared dataframe (with legacy column names) for helper methods
                self._write_general_info_sheet(writer, material_code, material_data_for_sheets)
                self._write_purchase_data_sheet(writer, material_code, material_data_for_sheets)
                self._write_anomaly_sheet(writer, material_code, material_data_for_sheets)
                self._write_periodicity_sheet(writer, material_code, material_data_for_sheets)
                self._write_risk_assessment_sheet(writer, material_code, material_data_for_sheets)
        except Exception as e:
            st.error(f"Ошибка при создании Excel файла для временного ряда {material_code}: {e}")
            from modules.utils import show_error_message
            show_error_message(e, f"Ошибка Excel для {material_code}", show_traceback=False)
            return None

        excel_buffer.seek(0)
        return excel_buffer.getvalue()

    def export_multiple_detailed_analysis(self, data, material_codes):
        """
        Экспортирует детальный анализ для НЕСКОЛЬКИХ временных рядов в один Excel файл.
        Каждый временной ряд получает свой набор листов с префиксом.

        Args:
            data: pandas DataFrame с обработанными данными
            material_codes: СПИСОК ID временных рядов для анализа
            
        Returns:
            bytes: содержимое Excel-файла или None, если нет выбранных рядов
        """
        if not material_codes:
            st.warning("Временные ряды для экспорта не выбраны.")
            return None

        excel_buffer = io.BytesIO()
        materials_processed = 0
        materials_skipped = 0
        
        try:
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_materials_to_process = len(material_codes)
                status_text.text(f"Подготовка отчета для {total_materials_to_process} временных рядов...")

                for i, material_code in enumerate(material_codes):
                    # Update progress
                    progress_value = (i + 1) / total_materials_to_process
                    progress_bar.progress(progress_value)
                    status_text.text(f"Обработка временного ряда: {material_code} ({i+1}/{total_materials_to_process})")
                    
                    # Resolve columns and filter data for the current time series
                    id_col, date_col, price_col = self._resolve_columns(data)
                    if id_col in data.columns:
                        material_data = data[data[id_col] == material_code].copy()
                    else:
                        material_data = pd.DataFrame()

                    # Sort by resolved date column if available
                    if date_col and date_col in material_data.columns:
                        material_data = material_data.sort_values(date_col)

                    if material_data.empty:
                        st.warning(f"Данные для временного ряда {material_code} не найдены, пропуск в Excel отчете.")
                        materials_skipped += 1
                        continue # Skip to the next
                    # Prepare columns for helper methods (rename to legacy names)
                    def _prepare_for_sheets(df):
                        df = df.copy()
                        if id_col and id_col in df.columns:
                            df = df.rename(columns={id_col: 'Материал'})
                        if date_col and date_col in df.columns:
                            df = df.rename(columns={date_col: 'ДатаСоздан'})
                        if price_col and price_col in df.columns:
                            df = df.rename(columns={price_col: 'Цена нетто (норм.)'})
                        return df

                    material_data_for_sheets = _prepare_for_sheets(material_data)

                    # Call helper methods to write sheets for this time series
                    self._write_general_info_sheet(writer, material_code, material_data_for_sheets)
                    self._write_purchase_data_sheet(writer, material_code, material_data_for_sheets)
                    self._write_anomaly_sheet(writer, material_code, material_data_for_sheets)
                    self._write_periodicity_sheet(writer, material_code, material_data_for_sheets)
                    self._write_risk_assessment_sheet(writer, material_code, material_data_for_sheets)
                    materials_processed += 1
                
                progress_bar.empty() # Remove progress bar on completion
                status_text.empty()

        except Exception as e:
            st.error(f"Критическая ошибка при создании общего Excel файла: {e}")
            from modules.utils import show_error_message
            show_error_message(e, f"Критическая ошибка Excel экспорта", show_traceback=True)
            return None

        if materials_processed == 0:
             st.error("Не удалось обработать ни один из выбранных временных рядов для Excel отчета.")
             return None

        if materials_skipped > 0:
            st.warning(f"{materials_skipped} временных рядов были пропущены из-за отсутствия данных.")
            
        excel_buffer.seek(0)
        return excel_buffer.getvalue()