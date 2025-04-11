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
            'suspicious_stable': 0.5,      # Подозрительно стабильная цена
            'price_jump': 200,             # Резкий скачок цены (%)
            'purchase_frequency': 3,       # Подозрительная частота закупок (дни)
            'end_period_activity': 3,      # Активность в конце периода (дни)
            'quarterly_increase': 25,      # Повышение в конце квартала (%)
            'round_price_tolerance': 0.01  # Допуск для округленных цен (%)
        }
    
    def analyze_security_risks(self, data, segments):
        """
        Проводит комплексный анализ безопасности данных - оптимизированная версия
        
        Args:
            data: pandas DataFrame с обработанными данными
            segments: словарь с сегментами материалов
        
        Returns:
            DataFrame с оценкой рисков для материалов
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
            unique_materials = data['Материал'].unique()
            total_materials = len(unique_materials)
            
            # Для очень больших объемов данных предлагаем оптимизированный анализ
            if (total_materials > 10000):
                limit_analysis = st.checkbox(
                    f"Обнаружено очень много материалов ({total_materials}). Ограничить анализ до 5000 случайных материалов?",
                    value=True
                )
                if limit_analysis:
                    # Используем numpy.random.Generator вместо numpy.random для лучшей производительности
                    rng = np.random.default_rng(42)  # Современная замена np.random.seed
                    unique_materials = rng.choice(unique_materials, size=5000, replace=False)
                    total_materials = 5000
                    st.info(f"Анализ ограничен до {total_materials} случайных материалов")
            
            # Используем пустой список для накопления результатов
            risk_data = []
            
            # Оптимизируем размер батча в зависимости от объема данных
            batch_size = min(1000, max(100, total_materials // 10))  # Увеличиваем размер для меньшего числа итераций
            
            # Создаем словарь для быстрого поиска сегмента - используем dict comprehension для оптимизации
            material_to_segment = {}
            for segment_name, segment_data in segments.items():
                material_to_segment.update({m: segment_name for m in segment_data['Материал'].values})
            
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
                    status_text.text(f"Анализ материалов... {i}/{total_materials}")
                
                # Оптимизируем фильтрацию данных для батча
                batch_data = data[data['Материал'].isin(batch_materials)]
                
                # Предварительная группировка по материалу для ускорения
                grouped_data = dict(list(batch_data.groupby('Материал')))
                
                # Обрабатываем каждый материал
                for material in batch_materials:
                    if material not in grouped_data:
                        continue
                        
                    material_data = grouped_data[material].sort_values('ДатаСоздан')
                    
                    if len(material_data) < 2:
                        continue
                    
                    # Вычисляем метрики риска
                    risk_metrics = self._calculate_risk_metrics(material_data)
                    
                    # Определяем категорию риска
                    risk_category, risk_factors = self._determine_risk_category(risk_metrics)
                    
                    # Находим сегмент материала
                    material_segment = material_to_segment.get(material, "Не определен")
                    
                    # Сохраняем только необходимые данные
                    risk_data.append({
                        'Материал': material,
                        'Категория риска': risk_category,
                        'Факторы риска': risk_factors,
                        'Сегмент': material_segment,
                        'Количество записей': len(material_data),
                        'Коэффициент вариации': risk_metrics['volatility'],
                        'Индекс аномальности цены': risk_metrics['price_anomaly_index'],
                        'Индекс дробления закупок': risk_metrics['purchase_fragmentation'],
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
                st.warning("Не удалось обнаружить материалы с признаками риска")
                return pd.DataFrame()
            
            # Отображаем результаты
            # self._display_security_analysis_results(risk_df) # УДАЛЕНО
            
            return risk_df
        
        except Exception as e:
            from modules.utils import show_error_message
            show_error_message(e, "Ошибка при анализе безопасности", show_traceback=True)
            st.info("Попробуйте ограничить количество анализируемых материалов или использовать меньший набор данных.")
            return pd.DataFrame()
    
    def _calculate_risk_metrics(self, material_data):
        """
        Вычисляет метрики риска для материала - оптимизированная версия
        
        Args:
            material_data: DataFrame с данными по одному материалу
                
        Returns:
            dict: словарь с метриками риска
        """
        metrics = {}
        
        # 1. Оптимизированный расчет базовых статистик
        prices = material_data['Цена нетто'].values  # Используем numpy array для ускорения
        
        # Вычисляем все статистики за один проход по массиву
        if len(prices) > 0:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            min_price = np.min(prices)
            max_price = np.max(prices)
            
            # 2. Волатильность цены (без создания промежуточных объектов)
            metrics['volatility'] = (std_price / mean_price) * 100 if mean_price > 0 else 0
            
            # 3. Индекс аномальности цены
            metrics['price_anomaly_index'] = (max_price / min_price) if min_price > 0 else 1
        else:
            # Для пустого массива устанавливаем значения по умолчанию
            metrics['volatility'] = 0
            metrics['price_anomaly_index'] = 1
        
        # 4. Анализ дробления закупок - оптимизированная версия
        dates = material_data['ДатаСоздан'].values  # Используем numpy array
        
        if len(dates) > 1:
            # Сортируем даты, только если они не отсортированы
            if not np.all(np.diff(dates) >= np.timedelta64(0)):
                dates = np.sort(dates)
                
            # Вычисляем разницу в днях
            date_diffs = np.diff(dates) / np.timedelta64(1, 'D')
            
            metrics['avg_days_between_purchases'] = np.mean(date_diffs) if date_diffs.size > 0 else 0
            small_intervals = np.sum(date_diffs <= self.risk_thresholds['purchase_frequency'])
            metrics['purchase_fragmentation'] = small_intervals / date_diffs.size * 100 if date_diffs.size > 0 else 0
        else:
            metrics['avg_days_between_purchases'] = 0
            metrics['purchase_fragmentation'] = 0
        
        # 5. Анализ сезонности - оптимизировано для быстродействия
        if len(material_data) >= 12:  # Минимум год данных для анализа сезонности
            # Используем numpy для расчетов вместо pandas где возможно
            months = material_data['ДатаСоздан'].dt.month.values
            unique_months = np.unique(months)
            
            if len(unique_months) > 1:
                # Для каждого уникального месяца находим среднюю цену
                monthly_means = np.array([np.mean(prices[months == month]) for month in unique_months])
                mean_of_means = np.mean(monthly_means)
                std_of_means = np.std(monthly_means)
                
                metrics['seasonal_deviation'] = (std_of_means / mean_of_means) * 100 if mean_of_means > 0 else 0
            else:
                metrics['seasonal_deviation'] = 0
        else:
            metrics['seasonal_deviation'] = 0
        
        # 6. Анализ активности в конце периодов
        # Избегаем создания копии DataFrame, используем исходные значения
        month_values = material_data['ДатаСоздан'].dt.month.values
        day_values = material_data['ДатаСоздан'].dt.day.values
        
        # Закупки в конце месяца
        end_of_month = np.mean(day_values >= 28) * 100
        
        # Закупки в конце квартала
        end_of_quarter_months = np.array([3, 6, 9, 12])  # Месяцы конца квартала
        is_quarter_end = np.isin(month_values, end_of_quarter_months)
        is_month_end = day_values >= 28
        end_of_quarter = np.mean(is_quarter_end & is_month_end) * 100 if len(is_quarter_end) > 0 else 0
        
        metrics['end_of_month_activity'] = end_of_month
        metrics['end_of_quarter_activity'] = end_of_quarter
        
        # 7. Проверка на округленные цены - векторизованная версия
        unique_prices = np.unique(prices)
        rounded_prices_count = 0
        
        # Оптимизация для разного количества цен
        if len(unique_prices) < 50:  # Для малого числа уникальных цен
            for price in unique_prices:
                if price <= 0:
                    continue
                for magnitude in [10, 100, 1000, 10000, 100000]:
                    if abs(price % magnitude) / price < self.risk_thresholds['round_price_tolerance']:
                        rounded_prices_count += 1
                        break
        else:
            # Массив для отслеживания уже учтенных цен
            processed = np.zeros(len(unique_prices), dtype=bool)
            
            # Проверяем для каждой величины
            for magnitude in [10, 100, 1000, 10000, 100000]:
                # Вычисляем маску только для положительных неучтенных цен
                valid_prices = (unique_prices > 0) & (~processed)
                if not np.any(valid_prices):
                    break
                    
                # Вычисляем относительную разницу от кратного значения
                valid_indices = np.where(valid_prices)[0]
                valid_price_values = unique_prices[valid_indices]
                
                # Векторизованная проверка на округленность
                remainder_ratio = np.abs(valid_price_values % magnitude) / valid_price_values
                rounded_mask = remainder_ratio < self.risk_thresholds['round_price_tolerance']
                
                # Подсчитываем и отмечаем учтенные цены
                rounded_indices = valid_indices[rounded_mask]
                rounded_prices_count += len(rounded_indices)
                processed[rounded_indices] = True
        
        unique_price_count = len(unique_prices)
        metrics['rounded_prices_ratio'] = rounded_prices_count / unique_price_count * 100 if unique_price_count > 0 else 0
        
        # 8. Вычисляем общий индекс подозрительности
        suspicion_index = (
            min(metrics['volatility'] * 0.5, 50) +                 # Волатильность (до 50 пунктов)
            min((metrics['price_anomaly_index'] - 1) * 10, 30) +    # Аномальность цены (до 30 пунктов)
            min(metrics['purchase_fragmentation'] * 0.5, 30) +      # Дробление закупок (до 30 пунктов)
            min(metrics['end_of_quarter_activity'] * 0.5, 20) +     # Активность в конце квартала (до 20 пунктов)
            min(metrics['rounded_prices_ratio'] * 0.3, 20)          # Округленные цены (до 20 пунктов)
        )
        
        metrics['suspicion_index'] = min(suspicion_index, 100)  # Максимум 100 пунктов
        
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
            risk_factors.append("Высокая волатильность цен")
        
        if risk_metrics['price_anomaly_index'] > 3:
            risk_factors.append("Значительные скачки цен")
        
        if risk_metrics['purchase_fragmentation'] > 30:
            risk_factors.append("Признаки дробления закупок")
        
        if risk_metrics['end_of_quarter_activity'] > 40:
            risk_factors.append("Повышенная активность в конце кварталов")
        
        if risk_metrics['rounded_prices_ratio'] > 70:
            risk_factors.append("Подозрительно округленные цены")
        
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
        # 1. Общая статистика
        st.subheader("Общая статистика анализа безопасности")
        
        total_materials = len(risk_df)
        high_risk = (risk_df['Категория риска'] == 'Высокий').sum()
        medium_risk = (risk_df['Категория риска'] == 'Средний').sum()
        low_risk = (risk_df['Категория риска'] == 'Низкий').sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Материалы с высоким риском", f"{high_risk} ({high_risk/total_materials*100:.1f}%)")
        
        with col2:
            st.metric("Материалы со средним риском", f"{medium_risk} ({medium_risk/total_materials*100:.1f}%)")
        
        with col3:
            st.metric("Материалы с низким риском", f"{low_risk} ({low_risk/total_materials*100:.1f}%)")
        
        # 2. Распределение рисков по сегментам
        st.subheader("Распределение рисков по сегментам")
        
        # Создаем сводную таблицу рисков по сегментам
        risk_by_segment = pd.crosstab(
            risk_df['Сегмент'], 
            risk_df['Категория риска'],
            normalize='index'
        ) * 100
        
        # Преобразуем для визуализации
        risk_by_segment_plot = risk_by_segment.reset_index().melt(
            id_vars=['Сегмент'],
            var_name='Категория риска',
            value_name='Процент материалов'
        )
        
        # Создаем столбчатую диаграмму
        fig = px.bar(
            risk_by_segment_plot,
            x='Сегмент',
            y='Процент материалов',
            color='Категория риска',
            title='Распределение категорий риска по сегментам',
            color_discrete_map={
                'Высокий': '#FF4B4B',
                'Средний': '#FFA72B',
                'Низкий': '#2ECC71'
            }
        )
        
        fig.update_layout(
            xaxis_title='Сегмент',
            yaxis_title='Процент материалов (%)',
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Топ материалов с высоким риском
        st.subheader("Топ материалов с высоким индексом подозрительности")
        
        # Фильтруем материалы с высоким риском
        # Убедимся, что risk_df передан и не пуст
        if risk_df is None or risk_df.empty:
             st.warning("Нет данных для отображения топа материалов с риском.")
             high_risk_materials = pd.DataFrame() # Пустой DataFrame
        else:
             high_risk_materials = risk_df.sort_values('Индекс подозрительности', ascending=False).head(20)

        # Создаем таблицу с подсветкой (только если есть материалы)
        if not high_risk_materials.empty:
             for index, row in high_risk_materials.iterrows():
                 with st.container():
                     col1, col2, col3 = st.columns([3, 2, 5])
                     
                     with col1:
                         st.write(f"**Материал:** {row['Материал']}")
                         st.write(f"**Сегмент:** {row['Сегмент']}")
                     
                     with col2:
                         # Цветовая индикация риска
                         risk_color = "#FF4B4B" if row['Категория риска'] == "Высокий" else "#FFA72B" if row['Категория риска'] == "Средний" else "#2ECC71"
                         st.markdown(f"""
                         <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; color: white;">
                             <strong>Риск:</strong> {row['Категория риска']}<br>
                             <strong>Индекс:</strong> {row['Индекс подозрительности']:.1f}
                         </div>
                         """, unsafe_allow_html=True)
                     
                     with col3:
                         st.markdown(f"**Факторы риска:** {row['Факторы риска']}")
                     
                     st.divider()
        
        # 4. Фильтр для поиска конкретных материалов
        st.subheader("Поиск материалов по уровню риска")
        
        # Убедимся, что risk_df передан и не пуст для фильтрации
        if risk_df is None or risk_df.empty:
            st.warning("Нет данных для фильтрации и экспорта.")
            filtered_risk_df = pd.DataFrame() # Пустой DataFrame
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                risk_category_filter = st.selectbox(
                    "Категория риска:",
                    ["Все", "Высокий", "Средний", "Низкий"],
                    key='security_risk_category_filter' # Добавляем ключ для уникальности
                )
            
            with col2:
                segment_filter = st.selectbox(
                    "Сегмент:",
                    ["Все"] + list(risk_df['Сегмент'].unique()),
                    key='security_segment_filter' # Добавляем ключ для уникальности
                )
            
            # Применяем фильтры
            filtered_risk_df = risk_df.copy()
            
            if risk_category_filter != "Все":
                filtered_risk_df = filtered_risk_df[risk_category_filter == filtered_risk_df['Категория риска']]
            
            if segment_filter != "Все":
                filtered_risk_df = filtered_risk_df[filtered_risk_df['Сегмент'] == segment_filter]

            # Отображаем отфильтрованные данные
            from modules.utils import create_styled_dataframe
            st.dataframe(
                create_styled_dataframe(
                    filtered_risk_df,
                    highlight_cols=['Индекс подозрительности'],
                    highlight_threshold=70,
                    precision=2
                ),
                use_container_width=True,
                height=500
            )
        
        # 5. Экспорт данных
        # Кнопки экспорта показываем только если есть отфильтрованные данные
        if not filtered_risk_df.empty:
            buffer = io.BytesIO()
            filtered_risk_df.to_csv(buffer, index=False, encoding='utf-8-sig')
            buffer.seek(0)
            st.download_button(
                label="Скачать отчет о рисках (CSV)",
                data=buffer,
                file_name="security_risk_report.csv",
                mime="text/csv; charset=utf-8-sig"
            )
            
            # Добавляем кнопку для скачивания Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                filtered_risk_df.to_excel(writer, sheet_name='Отчет о рисках', index=False)
                
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
                label="Скачать отчет о рисках (Excel)",
                data=excel_buffer,
                file_name="security_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def highlight_suspicious_materials(self, data, material_code=None):
        """
        Визуализирует подозрительные паттерны для конкретного материала
        
        Args:
            data: pandas DataFrame с обработанными данными
            material_code: код материала для анализа (если None, будет предложен выбор)
        """
        st.subheader("Детальный анализ подозрительных паттернов")
        
        # Если код материала не указан, предлагаем выбор
        if material_code is None:
            # Получаем список материалов с наибольшей волатильностью
            volatile_materials = data.groupby('Материал')['Цена нетто'].agg(['mean', 'std']).reset_index()
            volatile_materials['volatility'] = volatile_materials['std'] / volatile_materials['mean'] * 100
            volatile_materials = volatile_materials.sort_values('volatility', ascending=False).head(50)
            
            material_code = st.selectbox(
                "Выберите материал для анализа:",
                volatile_materials['Материал'].tolist()
            )
        
        # Фильтруем данные по выбранному материалу
        material_data = data[data['Материал'] == material_code].sort_values('ДатаСоздан')
        
        if material_data.empty:
            st.warning(f"Данные для материала {material_code} не найдены")
            return
        
        # Отображаем основную информацию о материале
        st.write(f"**Материал:** {material_code}")
        st.write(f"**Количество записей:** {len(material_data)}")
        st.write(f"**Период данных:** {material_data['ДатаСоздан'].min().strftime('%d.%m.%Y')} - {material_data['ДатаСоздан'].max().strftime('%d.%m.%Y')}")
        
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
        
        # Визуализация 1: График изменения цены со временем с подсветкой аномалий
        st.subheader("Динамика цен с подсветкой аномалий")
        
        # Вычисляем скользящее среднее и стандартное отклонение
        material_data['rolling_mean'] = material_data['Цена нетто'].rolling(window=5, min_periods=1).mean()
        material_data['rolling_std'] = material_data['Цена нетто'].rolling(window=5, min_periods=1).std()
        
        # Определяем верхнюю и нижнюю границы для аномалий (2 стандартных отклонения)
        material_data['upper_bound'] = material_data['rolling_mean'] + 2 * material_data['rolling_std']
        material_data['lower_bound'] = material_data['rolling_mean'] - 2 * material_data['rolling_std']
        
        # Помечаем аномалии
        material_data['is_anomaly'] = (material_data['Цена нетто'] > material_data['upper_bound']) | \
                                       (material_data['Цена нетто'] < material_data['lower_bound'])
        
        # Создаем график
        fig = go.Figure()
        
        # Добавляем линию цены
        fig.add_trace(go.Scatter(
            x=material_data['ДатаСоздан'],
            y=material_data['Цена нетто'],
            mode='lines+markers',
            name='Цена',
            line=dict(color='#1976D2', width=2)
        ))
        
        # Добавляем скользящее среднее
        fig.add_trace(go.Scatter(
            x=material_data['ДатаСоздан'],
            y=material_data['rolling_mean'],
            mode='lines',
            name='Скользящее среднее',
            line=dict(color='#2ECC71', width=1, dash='dash')
        ))
        
        # Добавляем границы аномалий
        fig.add_trace(go.Scatter(
            x=material_data['ДатаСоздан'],
            y=material_data['upper_bound'],
            mode='lines',
            name='Верхняя граница',
            line=dict(color='#FF7043', width=1, dash='dot'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=material_data['ДатаСоздан'],
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
                x=anomalies['ДатаСоздан'],
                y=anomalies['Цена нетто'],
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
            title=f'Динамика цен материала {material_code} с подсветкой аномалий',
            xaxis_title='Дата',
            yaxis_title='Цена нетто',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 2: Анализ периодичности закупок
        st.subheader("Анализ периодичности закупок")
        
        # Вычисляем разницу между последовательными датами
        material_data['days_diff'] = material_data['ДатаСоздан'].diff().dt.days
        
        # Создаем гистограмму интервалов между закупками
        fig = px.histogram(
            material_data.dropna(),
            x='days_diff',
            nbins=30,
            title='Распределение интервалов между закупками (дни)',
            labels={'days_diff': 'Интервал (дни)', 'count': 'Количество случаев'},
            color_discrete_sequence=['#1976D2']
        )
        
        # Добавляем вертикальную линию для порогового значения дробления закупок
        fig.add_vline(
            x=self.risk_thresholds['purchase_frequency'],
            line_dash="dash",
            line_color="#FF4B4B",
            annotation_text="Порог дробления закупок",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 3: Активность по месяцам и кварталам
        st.subheader("Активность по месяцам и кварталам")
        
        # Добавляем информацию о месяце, квартале и годе
        material_data['Год'] = material_data['ДатаСоздан'].dt.year
        material_data['Месяц'] = material_data['ДатаСоздан'].dt.month
        material_data['Квартал'] = material_data['ДатаСоздан'].dt.quarter
        
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
            colorbar=dict(title="Количество закупок")
        ))
        
        fig.update_layout(
            title='Тепловая карта активности закупок по месяцам',
            xaxis_title='Месяц',
            yaxis_title='Год',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Визуализация 4: Активность по дням месяца
        monthly_day_activity = material_data.groupby(['День']).size().reset_index(name='Количество')
        
        fig = px.bar(
            monthly_day_activity,
            x='День',
            y='Количество',
            title='Распределение закупок по дням месяца',
            labels={'День': 'День месяца', 'Количество': 'Количество закупок'},
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
        
        # Визуализация 5: Распределение цен (проверка на округленность)
        st.subheader("Анализ распределения цен")
        
        # Создаем гистограмму цен
        fig = px.histogram(
            material_data,
            x='Цена нетто',
            nbins=30,
            title='Распределение цен',
            labels={'Цена нетто': 'Цена', 'count': 'Количество случаев'},
            color_discrete_sequence=['#1976D2']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Проверка на округленные цены
        unique_prices = material_data['Цена нетто'].unique()
        rounded_prices = []
        
        for price in unique_prices:
            for magnitude in [10, 100, 1000, 10000, 100000]:
                if abs(price % magnitude) / price < self.risk_thresholds['round_price_tolerance']:
                    rounded_prices.append({
                        'Цена': price,
                        'Округлено до': magnitude
                    })
                    break
        
        if rounded_prices:
            st.subheader("Обнаруженные округленные цены")
            st.write(f"**Процент округленных цен:** {len(rounded_prices) / len(unique_prices) * 100:.1f}%")
            from modules.utils import format_streamlit_dataframe
            st.dataframe(
                format_streamlit_dataframe(pd.DataFrame(rounded_prices)),
                use_container_width=True,
                height=300  # Фиксированная высота для лучшего отображения
            )
            
        # Добавляем кнопку для экспорта детального анализа в Excel
        excel_data = self.export_detailed_analysis(data, material_code)
        if excel_data is not None:
            st.download_button(
                label="Скачать детальный анализ (Excel)",
                data=excel_data,
                file_name=f"security_analysis_{material_code}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def export_detailed_analysis(self, data, material_code):
        """
        Экспортирует детальный анализ подозрительного материала в Excel
        
        Args:
            data: pandas DataFrame с обработанными данными
            material_code: код материала для анализа
            
        Returns:
            bytes: содержимое Excel-файла
        """
        # Фильтруем данные по выбранному материалу
        material_data = data[data['Материал'] == material_code].sort_values('ДатаСоздан')
        
        if material_data.empty:
            return None
        
        # Создаем буфер для Excel
        excel_buffer = io.BytesIO()
        
        # Создаем Excel с несколькими листами
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # 1. Основная информация о материале
            material_info = pd.DataFrame({
                'Параметр': [
                    'Материал', 
                    'Количество записей', 
                    'Первая дата', 
                    'Последняя дата',
                    'Средняя цена',
                    'Минимальная цена',
                    'Максимальная цена',
                    'Стандартное отклонение',
                    'Коэффициент вариации (%)'
                ],
                'Значение': [
                    material_code,
                    len(material_data),
                    material_data['ДатаСоздан'].min().strftime('%d.%m.%Y'),
                    material_data['ДатаСоздан'].max().strftime('%d.%m.%Y'),
                    f"{material_data['Цена нетто'].mean():.2f}",
                    f"{material_data['Цена нетто'].min():.2f}",
                    f"{material_data['Цена нетто'].max():.2f}",
                    f"{material_data['Цена нетто'].std():.2f}",
                    f"{(material_data['Цена нетто'].std() / material_data['Цена нетто'].mean() * 100):.2f}%"
                ]
            })
            
            material_info.to_excel(writer, sheet_name='Общая информация', index=False)
            
            # Форматирование
            worksheet = writer.sheets['Общая информация']
            header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
            for col_num, value in enumerate(material_info.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(0, 0, 25)
            worksheet.set_column(1, 1, 30)
            
            # 2. Детальные данные о закупках
            material_data.to_excel(writer, sheet_name='Данные закупок', index=False)
            
            # Форматирование
            worksheet = writer.sheets['Данные закупок']
            for col_num, value in enumerate(material_data.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Автоподбор ширины столбцов
            for i, col in enumerate(material_data.columns):
                column_width = max(
                    material_data[col].astype(str).map(len).max(), 
                    len(col)
                ) + 2
                worksheet.set_column(i, i, min(column_width, 30))  # Не более 30 символов ширины
                
            # 3. Анализ аномалий
            
            # Вычисляем скользящее среднее и стандартное отклонение
            material_data_copy = material_data.copy()
            material_data_copy['rolling_mean'] = material_data_copy['Цена нетто'].rolling(window=5, min_periods=1).mean()
            material_data_copy['rolling_std'] = material_data_copy['Цена нетто'].rolling(window=5, min_periods=1).std()
            
            # Определяем верхнюю и нижнюю границы для аномалий (2 стандартных отклонения)
            material_data_copy['upper_bound'] = material_data_copy['rolling_mean'] + 2 * material_data_copy['rolling_std']
            material_data_copy['lower_bound'] = material_data_copy['rolling_mean'] - 2 * material_data_copy['rolling_std']
            
            # Помечаем аномалии
            material_data_copy['is_anomaly'] = (
                (material_data_copy['Цена нетто'] > material_data_copy['upper_bound']) | 
                (material_data_copy['Цена нетто'] < material_data_copy['lower_bound'])
            )
            
            # Фильтруем только аномалии
            anomalies = material_data_copy[material_data_copy['is_anomaly']]
            
            if not anomalies.empty:
                anomalies = anomalies[['ДатаСоздан', 'Цена нетто', 'rolling_mean', 'upper_bound', 'lower_bound']]
                anomalies.columns = ['Дата', 'Цена', 'Скользящее среднее', 'Верхняя граница', 'Нижняя граница']
                anomalies.to_excel(writer, sheet_name='Аномалии', index=False)
                
                # Форматирование
                worksheet = writer.sheets['Аномалии']
                for col_num, value in enumerate(anomalies.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Выделяем аномальные цены красным
                red_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
                for row_num, _ in enumerate(anomalies.index):
                    worksheet.write(row_num + 1, 1, anomalies.iloc[row_num, 1], red_format)
                    
                # Автоподбор ширины столбцов
                for i, col in enumerate(anomalies.columns):
                    column_width = max(
                        anomalies[col].astype(str).map(len).max(), 
                        len(col)
                    ) + 2
                    worksheet.set_column(i, i, min(column_width, 20))
            
            # 4. Периодичность закупок
            if len(material_data) > 1:
                material_data_copy['days_diff'] = material_data_copy['ДатаСоздан'].diff().dt.days
                
                periods_data = material_data_copy[['ДатаСоздан', 'days_diff']].dropna()  # Исправление ошибки dropна -> dropna
                periods_data.columns = ['Дата', 'Интервал (дни)']
                
                periods_data.to_excel(writer, sheet_name='Периодичность', index=False)
                
                # Форматирование
                worksheet = writer.sheets['Периодичность']
                for col_num, value in enumerate(periods_data.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Выделяем короткие интервалы оранжевым (возможное дробление закупок)
                orange_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
                for row_num, _ in enumerate(periods_data.index):
                    if periods_data.iloc[row_num, 1] <= self.risk_thresholds['purchase_frequency']:
                        worksheet.write(row_num + 1, 1, periods_data.iloc[row_num, 1], orange_format)
                        
                # Автоподбор ширины столбцов
                worksheet.set_column(0, 0, 20)
                worksheet.set_column(1, 1, 15)
            
            # 5. Расчет факторов риска
            risk_metrics = self._calculate_risk_metrics(material_data)
            risk_category, risk_factors = self._determine_risk_category(risk_metrics)
            
            risk_data = pd.DataFrame({
                'Показатель': [
                    'Волатильность цены (%)',
                    'Индекс аномальности цены',
                    'Индекс дробления закупок',
                    'Активность в конце месяца (%)',
                    'Активность в конце квартала (%)',
                    'Доля округленных цен (%)',
                    'Общий индекс подозрительности',
                    'Категория риска',
                    'Факторы риска'
                ],
                'Значение': [
                    f"{risk_metrics['volatility']:.2f}%",
                    f"{risk_metrics['price_anomaly_index']:.2f}",
                    f"{risk_metrics['purchase_fragmentation']:.2f}%",
                    f"{risk_metrics['end_of_month_activity']:.2f}%",
                    f"{risk_metrics['end_of_quarter_activity']:.2f}%",
                    f"{risk_metrics['rounded_prices_ratio']:.2f}%",
                    f"{risk_metrics['suspicion_index']:.2f}",
                    risk_category,
                    risk_factors
                ]
            })
            
            risk_data.to_excel(writer, sheet_name='Оценка рисков', index=False)
            
            # Форматирование
            worksheet = writer.sheets['Оценка рисков']
            for col_num, value in enumerate(risk_data.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Подсветка категории риска
            risk_color_format = None
            if risk_category == "Высокий":
                risk_color_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            elif risk_category == "Средний":
                risk_color_format = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
            else:
                risk_color_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
                
            worksheet.write(7, 1, risk_category, risk_color_format)
            
            # Автоподбор ширины столбцов
            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, 1, 50)
        
        excel_buffer.seek(0)
        return excel_buffer.getvalue()