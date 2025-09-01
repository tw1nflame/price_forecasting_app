import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64

class ForecastPreparation:
    """
    Класс для подготовки данных к прогнозированию
    """
    def __init__(self, role_names=None):
        # role_names should be the centralized ROLE_NAMES dict from app
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
    
    def segment_materials(self, data, min_data_points=24, max_volatility=30, min_activity_days=365):
        """
        Сегментирует временные ряды на основе их пригодности для различных методов прогнозирования
        
        Args:
            data: pandas DataFrame с обработанными данными
            min_data_points: минимальное количество точек данных для ML-прогнозирования
            max_volatility: максимальный коэффициент вариации для ML-прогнозирования
            min_activity_days: минимальное количество дней активности
        
        Returns:
            dict: сегменты временных рядов и статистика по сегментам
        """
        # Показываем прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Начинаем сегментацию временных рядов...")
        
        try:
            # Шаг 1: Вычисляем агрегированные метрики для каждого временного ряда - ОДИН РАЗ
            status_text.text("Вычисляем метрики для временных рядов...")
            
            # Получаем уникальные временные ряды для оценки прогресса
            if self.ROLE_ID not in data.columns:
                raise KeyError(f"Data must contain identifier column: '{self.ROLE_ID}'")
            unique_materials = data[self.ROLE_ID].unique()
            num_materials = len(unique_materials)
            
            # Используем более эффективные методы агрегации
            # Группируем данные один раз и вычисляем все необходимые метрики
            # Resolve date and price columns (prefer normalized price)
            date_col = self.ROLE_DATE if self.ROLE_DATE in data.columns else None
            norm_price_col = f"{self.ROLE_TARGET} (норм.)"
            price_col = None
            if norm_price_col in data.columns:
                price_col = norm_price_col
            elif self.ROLE_TARGET in data.columns:
                price_col = self.ROLE_TARGET
            else:
                raise KeyError(f"Data must contain price column: '{norm_price_col}' or '{self.ROLE_TARGET}'")

            material_metrics = data.groupby(self.ROLE_ID).agg(
                record_count=(self.ROLE_ID, 'count'),
                first_date=(date_col, 'min') if date_col is not None else (self.ROLE_ID, 'count'),
                last_date=(date_col, 'max') if date_col is not None else (self.ROLE_ID, 'count'),
                mean_price=(price_col, 'mean'),
                std_price=(price_col, 'std')
            ).reset_index()
            
            progress_bar.progress(0.2)
            status_text.text("Вычисляем дополнительные метрики...")
            
            # Вычисляем дополнительные метрики
            # Текущая дата (максимальная дата в данных)
            if date_col is None:
                current_date = None
            else:
                current_date = data[date_col].max()
            
            # Быстрое вычисление временного диапазона
            material_metrics['time_range'] = (material_metrics['last_date'] - material_metrics['first_date']).dt.days
            
            # Дни с последней активности
            material_metrics['days_since_last_activity'] = (current_date - material_metrics['last_date']).dt.days
            
            # Вычисляем коэффициент вариации (только если среднее не равно нулю)
            material_metrics['volatility'] = 0.0  # По умолчанию 0
            
            # Применяем векторизованные операции вместо циклов
            non_zero_mean_mask = material_metrics['mean_price'] != 0
            material_metrics.loc[non_zero_mean_mask, 'volatility'] = (
                material_metrics.loc[non_zero_mean_mask, 'std_price'] / 
                material_metrics.loc[non_zero_mean_mask, 'mean_price']
            ) * 100
            
            # Вычисляем стабильность значений (частоту повторения)
            status_text.text("Вычисляем стабильность значений...")
            progress_bar.progress(0.3)
            
            # Создаем отдельный массив для хранения данных о стабильности
            stability_data = []
            
            # Берем небольшое количество временных рядов для обработки за раз
            batch_size = 1000
            num_batches = (len(unique_materials) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(unique_materials))
                batch_materials = unique_materials[batch_start:batch_end]
                
                # Фильтруем данные только для текущего пакета
                batch_data = data[data[self.ROLE_ID].isin(batch_materials)]
                
                # Для каждого ряда в пакете
                for material in batch_materials:
                    material_prices = batch_data[batch_data[self.ROLE_ID] == material][price_col]
                    
                    # Если ряд имеет более одной записи
                    if len(material_prices) > 1:
                        # Вычисляем частоту наиболее часто встречающегося значения
                        value_counts = material_prices.value_counts()
                        most_common_count = value_counts.iloc[0]
                        is_stable = most_common_count / len(material_prices) >= 0.8
                    else:
                        # Если только одна запись, считаем значение стабильным
                        is_stable = True
                    
                    stability_data.append({
                        self.ROLE_ID: material,
                        'is_stable': is_stable
                    })
                
                # Обновляем прогресс-бар
                progress_value = 0.3 + (0.2 * (batch_idx + 1) / num_batches)
                progress_bar.progress(progress_value)
                status_text.text(f"Вычисляем стабильность значений... Обработано {batch_end} из {len(unique_materials)} временных рядов")
            
            # Создаем DataFrame из данных о стабильности
            stability_df = pd.DataFrame(stability_data)

            # Объединяем с основными метриками
            material_metrics = material_metrics.merge(stability_df, on=self.ROLE_ID, how='left')
            
            progress_bar.progress(0.5)
            status_text.text("Сегментируем временные ряды...")
            
            # Шаг 2: Создаем сегменты на основе вычисленных метрик
            segments = {}
            
            # Определяем сегменты с помощью векторизованных операций
            # 1. Неактивные
            inactive_mask = material_metrics['days_since_last_activity'] > min_activity_days
            inactive_materials = material_metrics[inactive_mask][self.ROLE_ID].tolist()
            
            # 2. Недостаточно данных
            insufficient_history_mask = (
                ~inactive_mask & 
                (material_metrics['record_count'] < 5)
            )
            insufficient_history_materials = material_metrics[insufficient_history_mask][self.ROLE_ID].tolist()
            
            # 3. Постоянное значение
            constant_price_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                (material_metrics['volatility'] < 1)
            )
            constant_price_materials = material_metrics[constant_price_mask][self.ROLE_ID].tolist()
            
            # 4. Высокая волатильность
            high_volatility_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                (material_metrics['volatility'] > max_volatility)
            )
            high_volatility_materials = material_metrics[high_volatility_mask][self.ROLE_ID].tolist()
            
            # 5. Подходит для прогнозирования
            ml_forecasting_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                ~high_volatility_mask & 
                (material_metrics['record_count'] >= min_data_points) & 
                (material_metrics['time_range'] >= 30)
            )
            ml_forecasting_materials = material_metrics[ml_forecasting_mask][self.ROLE_ID].tolist()
            
            # 6. Недостаточно активный
            naive_forecasting_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                ~high_volatility_mask & 
                ~ml_forecasting_mask & 
                (material_metrics['record_count'] >= 5)
            )
            naive_forecasting_materials = material_metrics[naive_forecasting_mask][self.ROLE_ID].tolist()
            
            progress_bar.progress(0.7)
            status_text.text("Создаем DataFrame для каждого сегмента...")
            
            # Шаг 3: Создаем DataFrame для каждого сегмента
            # Эффективный способ создания сегментов - фильтрация по спискам ID
            # Создаем словарь для маппинга ID в сегменты
            material_to_segment = {}
            
            for material in ml_forecasting_materials:
                material_to_segment[material] = 'Подходит для прогнозирования'
                
            for material in naive_forecasting_materials:
                material_to_segment[material] = 'Недостаточно активный'
                
            for material in constant_price_materials:
                material_to_segment[material] = 'Постоянное значение'
                
            for material in inactive_materials:
                material_to_segment[material] = 'Неактивные'
                
            for material in insufficient_history_materials:
                material_to_segment[material] = 'Недостаточно данных'
                
            for material in high_volatility_materials:
                material_to_segment[material] = 'Высокая волатильность'
            
            # Получаем уникальный набор временных рядов с метриками
            unique_materials_data = material_metrics.copy()

            # Добавляем колонку сегмента
            unique_materials_data['Сегмент'] = unique_materials_data[self.ROLE_ID].map(
                lambda x: material_to_segment.get(x, 'Не классифицирован')
            )
            
            progress_bar.progress(0.9)
            status_text.text("Подготавливаем результаты...")
            
            # Создаем словари для каждого сегмента
            segments = {
                'Подходит для прогнозирования': unique_materials_data[unique_materials_data['Сегмент'] == 'Подходит для прогнозирования'],
                'Недостаточно активный': unique_materials_data[unique_materials_data['Сегмент'] == 'Недостаточно активный'],
                'Постоянное значение': unique_materials_data[unique_materials_data['Сегмент'] == 'Постоянное значение'],
                'Неактивные': unique_materials_data[unique_materials_data['Сегмент'] == 'Неактивные'],
                'Недостаточно данных': unique_materials_data[unique_materials_data['Сегмент'] == 'Недостаточно данных'],
                'Высокая волатильность': unique_materials_data[unique_materials_data['Сегмент'] == 'Высокая волатильность']
            }
            
            # Переименовываем колонки для совместимости с остальным кодом
            column_mapping = {
                'record_count': 'Количество записей',
                'mean_price': 'Среднее значение',
                'std_price': 'Стд. отклонение',
                'volatility': 'Коэффициент вариации',
                'is_stable': 'Стабильное значение',
                'time_range': 'Временной диапазон',
                'days_since_last_activity': 'Дней с последней активности',
                'last_date': 'Последняя активность',
            }
            
            for segment_name, segment_df in segments.items():
                if not segment_df.empty:
                    # ensure identifier column present and then rename other columns for display
                    segments[segment_name] = segment_df.rename(columns=column_mapping)
            
            # Создаем статистику по сегментам
            stats = {
                'Подходит для прогнозирования': len(ml_forecasting_materials),
                'Недостаточно активный': len(naive_forecasting_materials),
                'Постоянное значение': len(constant_price_materials),
                'Неактивные': len(inactive_materials),
                'Недостаточно данных': len(insufficient_history_materials),
                'Высокая волатильность': len(high_volatility_materials)
            }
            
            progress_bar.progress(1.0)
            status_text.text("Сегментация завершена!")
            
            # Выводим статистику
            st.write("### Статистика по сегментам")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Подходит для прогнозирования", stats['Подходит для прогнозирования'])
                st.metric("Недостаточно активный", stats['Недостаточно активный'])
            
            with col2:
                st.metric("Постоянное значение", stats['Постоянное значение'])
                st.metric("Высокая волатильность", stats['Высокая волатильность'])
            
            with col3:
                st.metric("Неактивные", stats['Неактивные'])
                st.metric("Недостаточно данных", stats['Недостаточно данных'])
            
            # Общее количество временных рядов после сегментации
            total_materials = sum(stats.values())
            
            # Проверка на пересечение сегментов
            if total_materials != num_materials:
                st.warning(f"Внимание: общее количество временных рядов в сегментах ({total_materials}) "
                           f"не совпадает с количеством уникальных временных рядов ({num_materials})")
            
            return segments, stats
            
        except Exception as e:
            # В случае ошибки выводим информацию и возвращаем пустые словари
            st.error(f"Ошибка при сегментации: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return {}, {}
    
    def export_data_options(self, segments):
        """
        Предоставляет опции для экспорта данных по сегментам
        """
        st.subheader("Экспорт данных для прогнозирования")
        
        # Вкладки для разных типов экспорта
        export_tabs = st.tabs(["Экспорт по сегментам", "Массовый экспорт", "Настраиваемый экспорт"])
        
        with export_tabs[0]:
            # Экспорт отдельных сегментов
            self._export_by_segment(segments)
        
        with export_tabs[1]:
            # Массовый экспорт всех сегментов
            self._export_all_segments(segments)
        
        with export_tabs[2]:
            # Настраиваемый экспорт с фильтрацией
            self._export_custom(segments)
    
    def _export_by_segment(self, segments):
        """
        Экспорт данных отдельно по каждому сегменту
        """
        st.write("Выберите сегмент для экспорта и формат файла:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segment_to_export = st.selectbox(
                "Сегмент временных рядов:",
                list(segments.keys())
            )
        
        with col2:
            export_format = st.selectbox(
                "Формат экспорта:",
                ["Excel", "CSV"]
            )
        
        # Получаем данные выбранного сегмента
        segment_data = segments[segment_to_export]
        
        if not segment_data.empty:
            # Отображаем информацию о выбранном сегменте
            st.write(f"Выбран сегмент '{segment_to_export}' с {len(segment_data)} временными рядами.")
            
            # Опция включения подробной информации
            include_details = st.checkbox("Включить подробную информацию", value=True,
                                        help="Включает дополнительные статистики и метрики для каждого временного ряда")
            
            # Опция экспорта только ключевых колонок
            export_key_columns_only = st.checkbox("Экспортировать только ключевые колонки", value=False,
                                                help="Экспортирует только колонки, необходимые для прогнозирования")
            
            # Кнопка для экспорта
            if st.button("Экспортировать данные"):
                with st.spinner(f"Подготовка данных сегмента '{segment_to_export}'..."):
                    # Получаем полные данные для выбранных временных рядов
                    full_data = self._get_full_data_for_segment(segment_data, 
                                                              include_details=include_details,
                                                              key_columns_only=export_key_columns_only)
                    
                    # Экспортируем данные
                    if export_format == "CSV":
                        csv_data = self._export_to_csv(full_data)
                        file_name = f"{segment_to_export.lower().replace(' ', '_')}_data.csv"
                        
                        st.download_button(
                            label=f"Скачать {file_name}",
                            data=csv_data,
                            file_name=file_name,
                            mime="text/csv"
                        )
                        
                        st.success(f"Данные подготовлены! Нажмите кнопку выше, чтобы скачать {len(full_data)} записей.")
                    else:  # Excel
                        excel_data = self._export_to_excel(full_data, segment_to_export)
                        file_name = f"{segment_to_export.lower().replace(' ', '_')}_data.xlsx"
                        
                        st.download_button(
                            label=f"Скачать {file_name}",
                            data=excel_data,
                            file_name=file_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success(f"Данные подготовлены! Нажмите кнопку выше, чтобы скачать {len(full_data)} записей.")
        else:
            st.warning(f"Сегмент '{segment_to_export}' не содержит временных рядов.")
    
    def _export_all_segments(self, segments):
        """
        Экспортирует все сегменты в один ZIP-архив.
        """
        import zipfile
        import os
        import tempfile
        from datetime import datetime
        
        st.write("Массовый экспорт всех сегментов в отдельные файлы:")
        
        # Опции экспорта
        export_format = st.selectbox(
            "Формат экспорта для всех сегментов:",
            ["Excel", "CSV"],
            key="bulk_export_format"
        )
        
        # Опция включения подробной информации
        include_details = st.checkbox("Включить подробную информацию для всех сегментов", value=True,
                                    key="bulk_include_details")
        
        # Опция экспорта только ключевых колонок
        export_key_columns_only = st.checkbox("Экспортировать только ключевые колонки для всех сегментов", 
                                            value=False, key="bulk_key_columns")
        
        # Дополнительная опция для фильтрации пустых сегментов
        skip_empty_segments = st.checkbox("Пропустить пустые сегменты", value=True)
        
        # Ограничение количества временных рядов и строк данных
        max_materials = st.number_input("Максимальное количество временных рядов на сегмент", min_value=100, value=2000, step=100)
        max_rows = st.number_input("Максимальное количество строк данных на сегмент", min_value=1000, value=100000, step=1000)
        
        # Кнопка для экспорта всех сегментов
        if st.button("Экспортировать все сегменты"):
            # Общий прогресс-бар
            overall_progress = st.progress(0)
            status_text = st.empty()
            status_text.text("Подготовка к экспорту...")
            
            # Создаем временную директорию для файлов
            with tempfile.TemporaryDirectory() as temp_dir:
                # Подготавливаем файлы для каждого сегмента
                with st.spinner("Подготовка данных для всех сегментов..."):
                    # Создадим ZIP-архив
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"all_segments_{timestamp}.zip"
                    zip_path = os.path.join(temp_dir, zip_filename)
                    
                    # Фильтруем и сортируем сегменты
                    non_empty_segments = {name: data for name, data in segments.items() 
                                          if not (skip_empty_segments and data.empty)}
                    
                    # Сортируем сегменты по размеру (от меньшего к большему)
                    sorted_segments = sorted(non_empty_segments.items(), key=lambda x: len(x[1]))
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf: # Используем сжатие
                        
                        # Для каждого сегмента
                        segment_count = 0
                        total_segments = len(sorted_segments)
                        
                        for segment_idx, (segment_name, segment_data) in enumerate(sorted_segments):
                            # Обновляем прогресс
                            segment_progress = segment_idx / total_segments
                            overall_progress.progress(segment_progress)
                            status_text.text(f"Обработка сегмента {segment_name} ({segment_idx+1}/{total_segments})...")
                            
                            # Ограничиваем количество временных рядов для предотвращения переполнения памяти
                            limited_segment_data = segment_data
                            if len(segment_data) > max_materials:
                                limited_segment_data = segment_data.head(max_materials)
                                status_text.text(f"Ограничиваем сегмент {segment_name} до {max_materials} временных рядов...")
                            
                            try:
                                # Получаем полные данные для текущего сегмента
                                full_data = self._get_full_data_for_segment(
                                    limited_segment_data,
                                    include_details=include_details,
                                    key_columns_only=export_key_columns_only,
                                    max_rows=max_rows
                                )
                                
                                if full_data.empty:
                                    status_text.text(f"Сегмент {segment_name} не содержит данных, пропускаем...")
                                    continue
                                
                                # Создаем имя файла для сегмента
                                segment_filename = f"{segment_name.lower().replace(' ', '_')}_data"
                                file_path = os.path.join(temp_dir, segment_filename)
                                
                                # Экспортируем в соответствующий формат
                                status_text.text(f"Экспорт сегмента {segment_name} в {export_format}...")
                                
                                if export_format == "CSV":
                                    full_path = f"{file_path}.csv"
                                    full_data.to_csv(full_path, index=False)
                                else:  # Excel
                                    full_path = f"{file_path}.xlsx"
                                    
                                    # Используем оптимизированную версию экспорта в Excel
                                    with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
                                        sheet_name = segment_name[:31]  # Excel ограничивает имя листа 31 символом
                                        full_data.to_excel(writer, sheet_name=sheet_name, index=False)
                                        
                                        # Минимальное форматирование для ускорения
                                        workbook = writer.book
                                        worksheet = writer.sheets[sheet_name]
                                        
                                        # Формат для заголовков
                                        header_format = workbook.add_format({'bold': True, 'fg_color': '#D7E4BC'})
                                        for col_num, column in enumerate(full_data.columns):
                                            worksheet.write(0, col_num, column, header_format)
                                        
                                        # Автофильтр и закрепление строки
                                        worksheet.autofilter(0, 0, len(full_data), len(full_data.columns) - 1)
                                        worksheet.freeze_panes(1, 0)
                                
                                # Добавляем файл в архив
                                zipf.write(full_path, os.path.basename(full_path))
                                segment_count += 1
                                
                            except Exception as e:
                                status_text.text(f"Ошибка при обработке сегмента {segment_name}: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                continue
                    
                    # Обновляем прогресс
                    overall_progress.progress(1.0)
                    status_text.text("Архив создан, готовим к загрузке...")
                    
                    # Читаем созданный ZIP-файл
                    with open(zip_path, 'rb') as f:
                        zip_data = f.read()
                
                # Создаем кнопку для скачивания архива
                st.download_button(
                    label=f"Скачать архив с {segment_count} сегментами",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip"
                )
                
                st.success(f"Архив с данными подготовлен! Всего сегментов в архиве: {segment_count}")
                status_text.empty()  # Очищаем текст статуса
    
    def _export_custom(self, segments):
        """
        Настраиваемый экспорт с возможностью фильтрации. 
        Экспортирует отфильтрованные данные в один файл Excel или CSV.
        """
        st.write("Настраиваемый экспорт с фильтрацией временных рядов:")
        
        # Опции фильтрации
        st.subheader("Параметры фильтрации")
        
        # 1. Выбор сегментов для экспорта
        selected_segments = st.multiselect(
            "Выберите сегменты для экспорта:",
            list(segments.keys()),
            default=list(segments.keys())
        )
        
        # 2. Фильтры по характеристикам
        with st.expander("Дополнительные фильтры", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                min_records = st.number_input("Минимальное количество записей", min_value=1, value=5)
                max_volatility = st.slider("Максимальный коэффициент вариации (%)", 0, 100, 50)
            
            with col2:
                min_time_range = st.number_input("Минимальный временной диапазон (дни)", min_value=1, value=30)
                max_days_inactive = st.number_input("Максимальные дни неактивности", min_value=0, value=365)
            
            # Ограничение количества временных рядов и строк
            col1, col2 = st.columns(2)
            with col1:
                max_materials = st.number_input("Макс. количество временных рядов", min_value=10, value=1000, step=10)
            with col2:
                max_rows = st.number_input("Макс. количество строк", min_value=1000, value=100000, step=1000)
        
        # Опции экспорта
        st.subheader("Параметры экспорта")
        export_format = st.selectbox(
            "Формат экспорта:",
            ["Excel", "CSV"],
            key="custom_export_format"
        )
        
        include_details = st.checkbox("Включить подробную информацию", value=True,
                                    key="custom_include_details")
        
        export_key_columns_only = st.checkbox("Экспортировать только ключевые колонки", 
                                            value=False, key="custom_key_columns")
        
        # Кнопка для применения фильтров и экспорта
        if st.button("Применить фильтры и экспортировать"):
            # Прогресс-бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Применение фильтров...")
            
            try:
                # Объединяем данные из выбранных сегментов
                all_filtered_data = pd.DataFrame()
                
                # Подсчитываем общее количество временных рядов до фильтрации
                total_materials_before = sum(len(segments[name]) for name in selected_segments if name in segments)
                status_text.text(f"Анализ {total_materials_before} временных рядов из {len(selected_segments)} сегментов...")
                
                # Обрабатываем каждый сегмент
                materials_processed = 0
                for i, segment_name in enumerate(selected_segments):
                    # Проверяем наличие сегмента
                    if segment_name not in segments:
                        continue
                    
                    segment_data = segments[segment_name]
                    
                    # Пропускаем пустые сегменты
                    if segment_data.empty:
                        continue
                    
                    # Обновляем прогресс
                    segment_progress = i / len(selected_segments) * 0.5
                    progress_bar.progress(segment_progress)
                    status_text.text(f"Фильтрация сегмента {segment_name}...")
                    
                    # Применяем фильтры
                    filtered_segment = segment_data[
                        (segment_data['Количество записей'] >= min_records) &
                        (segment_data.get('Коэффициент вариации', 0) <= max_volatility) &
                        (segment_data.get('Временной диапазон', 0) >= min_time_range) &
                        (segment_data.get('Дней с последней активности', 0) <= max_days_inactive)
                    ]
                    
                    # Добавляем информацию о сегменте
                    if not filtered_segment.empty:
                        filtered_segment['Сегмент'] = segment_name
                        all_filtered_data = pd.concat([all_filtered_data, filtered_segment])
                    
                    # Увеличиваем счетчик обработанных
                    materials_processed += len(segment_data)
                    
                # Ограничиваем количество
                if len(all_filtered_data) > max_materials:
                    status_text.text(f"Ограничение количества временных рядов до {max_materials}...")
                    all_filtered_data = all_filtered_data.head(max_materials)
                
                progress_bar.progress(0.6)
                
                # Если после фильтрации остались данные
                if not all_filtered_data.empty:
                    status_text.text(f"Подготовка данных для экспорта ({len(all_filtered_data)} временных рядов)...")
                    
                    # Получаем полные данные для отфильтрованных рядов
                    full_data = self._get_full_data_for_segment(
                        all_filtered_data, 
                        include_details=include_details,
                        key_columns_only=export_key_columns_only,
                        max_rows=max_rows
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("Подготовка файла для экспорта...")
                    
                    # Добавляем колонку с сегментом, если она еще не добавлена
                    if 'Сегмент' in all_filtered_data.columns and 'Сегмент' not in full_data.columns:
                        material_to_segment_map = dict(zip(all_filtered_data[self.ROLE_ID], all_filtered_data['Сегмент']))
                        full_data['Сегмент'] = full_data[self.ROLE_ID].map(material_to_segment_map).fillna('Неизвестно')

                    # Сообщаем о размере данных
                    data_size_mb = full_data.memory_usage(deep=True).sum() / (1024 * 1024)
                    if data_size_mb > 100:
                        st.warning(f"Размер данных: {data_size_mb:.1f} МБ. Экспорт может занять некоторое время.")
                    
                    try:
                        # Экспортируем данные
                        progress_bar.progress(0.9)
                        status_text.text(f"Экспорт данных в формате {export_format}...")
                        
                        # --- Экспорт в один файл (Excel или CSV) ---
                        if export_format == "CSV":
                            file_data = self._export_to_csv(full_data)
                            file_name = "custom_filtered_data.csv"
                            mime_type = "text/csv; charset=utf-8-sig"
                        else:  # Excel
                            file_data = self._export_to_excel(full_data, "Отфильтрованные данные")
                            file_name = "custom_filtered_data.xlsx"
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            
                        st.download_button(
                            label=f"Скачать {file_name}",
                            data=file_data,
                            file_name=file_name,
                            mime=mime_type
                        )
                        # --- Конец экспорта в один файл ---

                        progress_bar.progress(1.0)
                        status_text.text("Экспорт завершен!")
                        
                    except Exception as e:
                        progress_bar.progress(1.0)
                        status_text.text("Ошибка при экспорте!")
                        from modules.utils import show_error_message
                        show_error_message(e, "Ошибка при экспорте данных", show_traceback=True)
                        
                        # Предлагаем альтернативный вариант экспорта
                        st.info("""
                        Попробуйте выполнить экспорт еще раз с меньшим объемом данных или в другом формате.
                        Для очень больших объемов данных рекомендуется использовать CSV формат вместо Excel.
                        """)
                    
                    st.success(f"Данные успешно отфильтрованы и подготовлены! "
                             f"Количество временных рядов после фильтрации: {all_filtered_data[self.ROLE_ID].nunique()}, "
                             f"всего записей: {len(full_data)}")
                    
                    # Показываем сводку по сегментам после фильтрации
                    if 'Сегмент' in all_filtered_data.columns:
                        segment_counts = all_filtered_data.groupby('Сегмент')[self.ROLE_ID].nunique().reset_index()
                        segment_counts.columns = ['Сегмент', 'Количество временных рядов']
                        
                        from modules.utils import format_streamlit_dataframe
                        st.dataframe(
                            format_streamlit_dataframe(segment_counts),
                            use_container_width=True,
                            height=400  # Фиксированная высота для лучшего отображения
                        )
                else:
                    progress_bar.progress(1.0)
                    status_text.text("Нет данных для экспорта!")
                    st.warning("После применения фильтров не осталось данных. Попробуйте смягчить критерии фильтрации.")
            
            except Exception as e:
                progress_bar.progress(1.0)
                status_text.text("Ошибка при экспорте!")
                st.error(f"Произошла ошибка при экспорте данных: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    def _get_full_data_for_segment(self, segment_data, include_details=True, key_columns_only=False, max_rows=100000):
        """
        Получает полные данные для временных рядов в сегменте (role-aware) и возвращает DataFrame
        Перед возвратом пытается переименовать колонки обратно в исходные названия, выбранные пользователем в маппинге.
        """
        # Показываем предупреждение о больших объемах данных
        if len(segment_data) > 1000:
            st.warning(f"Выбрано много временных рядов ({len(segment_data)}). Данные могут обрабатываться дольше обычного.")

        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Подготовка данных для экспорта...")

        # Получаем идентификатор колонки в segment_data (обычно это ROLE_ID)
        id_col = self.ROLE_ID if self.ROLE_ID in segment_data.columns else segment_data.columns[0]
        materials = segment_data[id_col].tolist()

        # Ограничиваем количество для предотвращения слишком длительной обработки
        if len(materials) > 5000:
            status_text.text(f"Слишком много временных рядов ({len(materials)}). Ограничиваем до 5000...")
            materials = materials[:5000]

        # Получаем полные данные из session_state
        if 'processed_data' not in st.session_state:
            return pd.DataFrame()

        full_data = st.session_state.processed_data
        # Определяем какую колонку идентификатора использовать в full_data
        full_id_col = self.ROLE_ID if self.ROLE_ID in full_data.columns else full_data.columns[0]

        progress_bar.progress(0.1)
        status_text.text("Индексирование данных...")

        materials_set = set(materials)

        # Эффективная фильтрация
        if len(full_data) > 100000:
            filtered_rows = []
            chunk_size = min(max(len(full_data) // 20, 10000), 50000)
            for i in range(0, len(full_data), chunk_size):
                chunk = full_data.iloc[i:i+chunk_size]
                if full_id_col in chunk.columns:
                    mask = chunk[full_id_col].apply(lambda x: x in materials_set)
                    filtered_chunk = chunk[mask]
                else:
                    # fallback: use isin on all columns (slower)
                    filtered_chunk = chunk[chunk.apply(lambda row: row.isin(materials_set).any(), axis=1)]
                filtered_rows.append(filtered_chunk)

                progress = 0.1 + 0.4 * (i + chunk_size) / len(full_data)
                progress_bar.progress(min(progress, 0.5))
                status_text.text(f"Фильтрация данных... {min((i + chunk_size) / len(full_data) * 100, 100):.1f}%")

            if filtered_rows:
                filtered_data = pd.concat(filtered_rows, ignore_index=True)
            else:
                filtered_data = pd.DataFrame(columns=full_data.columns)
        else:
            # Для меньших данных используем стандартный метод
            if full_id_col in full_data.columns:
                filtered_data = full_data[full_data[full_id_col].isin(materials)].copy()
            else:
                filtered_data = full_data[full_data.apply(lambda row: row.isin(materials_set).any(), axis=1)].copy()

            progress_bar.progress(0.5)
            status_text.text("Данные отфильтрованы...")

        # Ограничиваем количество строк для предотвращения переполнения памяти
        if len(filtered_data) > max_rows:
            status_text.text(f"Слишком много строк данных ({len(filtered_data)}). Ограничиваем до {max_rows}...")
            filtered_data = filtered_data.head(max_rows)

        # Если нужны только ключевые колонки
        if key_columns_only:
            canonical_key_columns = [full_id_col]
            if self.ROLE_DATE in filtered_data.columns:
                canonical_key_columns.append(self.ROLE_DATE)
            norm_col = f"{self.ROLE_TARGET} (норм.)"
            if norm_col in filtered_data.columns:
                canonical_key_columns.append(norm_col)
            # Валюта и курс
            currency_col = self.role_names.get('ROLE_CURRENCY')
            rate_col = self.role_names.get('ROLE_RATE')
            if currency_col and currency_col in filtered_data.columns:
                canonical_key_columns.append(currency_col)
            if rate_col and rate_col in filtered_data.columns:
                canonical_key_columns.append(rate_col)
            # Time parts
            for tc in ['Год', 'Месяц', 'День']:
                if tc in filtered_data.columns:
                    canonical_key_columns.append(tc)

            existing_key_columns = [col for col in canonical_key_columns if col in filtered_data.columns]
            filtered_data = filtered_data[existing_key_columns]

            progress_bar.progress(0.7)
            status_text.text("Выбраны ключевые колонки...")

        # Если нужно включить детальную информацию о временных рядах
        if include_details and not key_columns_only:
            status_text.text("Добавление детальной информации...")
            progress_bar.progress(0.6)

            detail_columns = [
                'Количество записей', 'Коэффициент вариации',
                'Стабильное значение', 'Временной диапазон',
                'Дней с последней активности', 'Неактивный временной ряд',
                'Среднее значение', 'Стд. отклонение'
            ]

            available_detail_columns = [col for col in detail_columns if col in segment_data.columns]
            if available_detail_columns:
                details_df = segment_data[[id_col] + available_detail_columns].drop_duplicates(id_col)
                # merge on appropriate id column name in filtered_data (full_id_col)
                if id_col == full_id_col:
                    filtered_data = pd.merge(filtered_data, details_df, on=id_col, how='left')
                else:
                    # align keys by renaming details df to full_id_col temporarily if possible
                    details_df = details_df.rename(columns={id_col: full_id_col})
                    filtered_data = pd.merge(filtered_data, details_df, on=full_id_col, how='left')

            progress_bar.progress(0.8)
            status_text.text("Детальная информация добавлена...")

        progress_bar.progress(1.0)
        status_text.text("Данные готовы для экспорта!")

        # Перед возвратом данных — попытка вернуть названия колонок к исходным, выбранным в маппинге
        try:
            if 'column_mapping' in st.session_state and st.session_state.get('column_mapping'):
                mapping = st.session_state['column_mapping'] or {}
                rename_map = {}
                # mapping keys are canonical role names like 'ID', 'Дата', 'Целевая Колонка'
                for canonical_role, original_col in mapping.items():
                    if not original_col:
                        continue
                    # если каноническое имя колонкой (например 'ID') присутствует — переименуем
                    if canonical_role in filtered_data.columns:
                        rename_map[canonical_role] = original_col
                    # normalized price column
                    canonical_norm = f"{canonical_role} (норм.)"
                    if canonical_norm in filtered_data.columns:
                        rename_map[canonical_norm] = f"{original_col} (норм.)"

                if rename_map:
                    filtered_data = filtered_data.rename(columns=rename_map)
        except Exception:
            # Если что-то пошло не так при переименовании, просто возвращаем канонический набор
            pass

        return filtered_data
    
    def _export_to_csv(self, data):
        """
        Экспортирует данные в формате CSV с правильной кодировкой
        """
        csv_buffer = io.BytesIO()  # Используем BytesIO вместо StringIO для работы с бинарными данными
        data.to_csv(csv_buffer, index=False, encoding='utf-8-sig')  # UTF-8 с BOM для Excel
        return csv_buffer.getvalue()
    
    def _export_to_excel(self, data, sheet_name="Данные"):
        """
        Экспортирует данные в формате Excel
        
        Args:
            data: DataFrame для экспорта
            sheet_name: имя листа в Excel-файле
            
        Returns:
            bytes: содержимое Excel-файла
        """
        excel_buffer = io.BytesIO()
        
        # Создаем writer
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Имя листа не должно превышать 31 символ (ограничение Excel)
            sheet_name = sheet_name[:31]
            
            # Записываем данные в Excel
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Получаем workbook и worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Добавляем форматирование (оптимизировано)
            
            # 1. Формат для заголовков
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # 2. Формат для числовых колонок
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            
            # 3. Автоподбор ширины столбцов (оптимизированный)
            # Используем только первые 100 строк для определения ширины
            sample_data = data.head(100) if len(data) > 100 else data
            
            # Применяем форматирование
            for col_num, column in enumerate(data.columns):
                # Применяем формат заголовка
                worksheet.write(0, col_num, column, header_format)
                
                # Устанавливаем ширину колонки
                try:
                    # Вычисляем ширину на основе заголовка и выборки данных
                    max_len = len(str(column))
                    
                    # Проверяем, есть ли данные
                    if not sample_data.empty:
                        # Для строковых колонок используем данные для определения ширины
                        if sample_data[column].dtype == 'object':
                            sample_width = sample_data[column].astype(str).str.len().max()
                            if not pd.isna(sample_width):
                                max_len = max(max_len, sample_width)
                        # Для числовых колонок используем фиксированную ширину
                        elif pd.api.types.is_numeric_dtype(sample_data[column]):
                            max_len = max(max_len, 12)  # Стандартная ширина для чисел
                            worksheet.set_column(col_num, col_num, None, number_format)
                        # Для дат используем фиксированную ширину
                        elif pd.api.types.is_datetime64_dtype(sample_data[column]):
                            max_len = max(max_len, 18)  # Стандартная ширина для дат
                    
                    # Ограничиваем максимальную ширину
                    column_width = min(max_len + 2, 30)  # не больше 30 символов
                    worksheet.set_column(col_num, col_num, column_width)
                except Exception:
                    # В случае ошибки устанавливаем стандартную ширину
                    worksheet.set_column(col_num, col_num, 15)
            
            # Добавляем автофильтр
            worksheet.autofilter(0, 0, len(data), len(data.columns) - 1)
            
            # Закрепляем первую строку
            worksheet.freeze_panes(1, 0)
        
        # Получаем значение из буфера
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    def prepare_for_forecasting(self, data, material, forecast_horizon=12):
        """
        Подготавливает данные для прогнозирования
        
        Args:
            data: pandas DataFrame с обработанными данными
            material: ID временного ряда для прогнозирования
            forecast_horizon: горизонт прогнозирования в месяцах
        
        Returns:
            dict: подготовленные данные для прогнозирования
        """
        # Получаем данные для указанного временного ряда
        material_data = data[data[self.ROLE_ID] == material].copy()
        
        # Проверяем, есть ли данные
        if material_data.empty:
            return None
        
        # Сортируем по дате
        material_data = material_data.sort_values(self.ROLE_DATE)
        
        # Вычисляем частоту данных
        date_diffs = material_data[self.ROLE_DATE].diff().dropna()
        
        # Если данных меньше 2, не можем определить частоту
        if len(date_diffs) < 1:
            return None
        
        # Определяем медианную частоту в днях
        median_freq_days = date_diffs.median().days
        
        # Определяем частоту данных (дневная, недельная, месячная и т.д.)
        if median_freq_days < 3:
            freq = 'D'  # Дневная
        elif median_freq_days < 10:
            freq = 'W'  # Недельная
        else:
            freq = 'M'  # Месячная
        
        # Создаем временной ряд с ресамплингом
        material_data.set_index(self.ROLE_DATE, inplace=True)
        
        # Ресамплинг данных
        resampled_data = material_data[f"{self.ROLE_TARGET} (норм.)"].resample(freq).mean()
        
        # Заполняем пропущенные значения
        resampled_data = resampled_data.interpolate(method='linear')
        
        # Определяем последнюю дату в данных
        last_date = resampled_data.index[-1]
        
        # Создаем датаframe для прогнозирования
        forecast_data = {
            'id': material,
            'time_series': resampled_data,
            'freq': freq,
            'last_date': last_date,
            'forecast_horizon': forecast_horizon
        }
        
        return forecast_data