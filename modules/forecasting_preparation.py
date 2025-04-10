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
    
    def __init__(self):
        pass
    
    def segment_materials(self, data, min_data_points=24, max_volatility=30, min_activity_days=365):
        """
        Сегментирует материалы на основе их пригодности для различных методов прогнозирования
        
        Args:
            data: pandas DataFrame с обработанными данными
            min_data_points: минимальное количество точек данных для ML-прогнозирования
            max_volatility: максимальный коэффициент вариации для ML-прогнозирования
            min_activity_days: минимальное количество дней активности
        
        Returns:
            dict: сегменты материалов и статистика по сегментам
        """
        # Показываем прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Начинаем сегментацию материалов...")
        
        try:
            # Шаг 1: Вычисляем агрегированные метрики для каждого материала - ОДИН РАЗ
            status_text.text("Вычисляем метрики для материалов...")
            
            # Получаем уникальные материалы для оценки прогресса
            unique_materials = data['Материал'].unique()
            num_materials = len(unique_materials)
            
            # Используем более эффективные методы агрегации
            # Группируем данные один раз и вычисляем все необходимые метрики
            material_metrics = data.groupby('Материал').agg(
                record_count=('Материал', 'count'),
                first_date=('ДатаСоздан', 'min'),
                last_date=('ДатаСоздан', 'max'),
                mean_price=('Цена нетто', 'mean'),
                std_price=('Цена нетто', 'std')
            ).reset_index()
            
            progress_bar.progress(0.2)
            status_text.text("Вычисляем дополнительные метрики...")
            
            # Вычисляем дополнительные метрики
            # Текущая дата (максимальная дата в данных)
            current_date = data['ДатаСоздан'].max()
            
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
            
            # Вычисляем стабильность цен (частоту повторения цен)
            status_text.text("Вычисляем стабильность цен...")
            progress_bar.progress(0.3)
            
            # Создаем отдельный массив для хранения данных о стабильности
            stability_data = []
            
            # Берем небольшое количество материалов для обработки за раз
            batch_size = 1000
            num_batches = (len(unique_materials) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(unique_materials))
                batch_materials = unique_materials[batch_start:batch_end]
                
                # Фильтруем данные только для текущего пакета материалов
                batch_data = data[data['Материал'].isin(batch_materials)]
                
                # Для каждого материала в пакете
                for material in batch_materials:
                    material_prices = batch_data[batch_data['Материал'] == material]['Цена нетто']
                    
                    # Если материал имеет более одной записи
                    if len(material_prices) > 1:
                        # Вычисляем частоту наиболее часто встречающегося значения
                        value_counts = material_prices.value_counts()
                        most_common_count = value_counts.iloc[0]
                        is_stable = most_common_count / len(material_prices) >= 0.8
                    else:
                        # Если только одна запись, считаем цену стабильной
                        is_stable = True
                    
                    stability_data.append({
                        'Материал': material,
                        'is_stable': is_stable
                    })
                
                # Обновляем прогресс-бар
                progress_value = 0.3 + (0.2 * (batch_idx + 1) / num_batches)
                progress_bar.progress(progress_value)
                status_text.text(f"Вычисляем стабильность цен... Обработано {batch_end} из {len(unique_materials)} материалов")
            
            # Создаем DataFrame из данных о стабильности
            stability_df = pd.DataFrame(stability_data)
            
            # Объединяем с основными метриками
            material_metrics = material_metrics.merge(stability_df, on='Материал', how='left')
            
            progress_bar.progress(0.5)
            status_text.text("Сегментируем материалы...")
            
            # Шаг 2: Создаем сегменты на основе вычисленных метрик
            segments = {}
            
            # Определяем сегменты с помощью векторизованных операций
            # 1. Неактивные материалы
            inactive_mask = material_metrics['days_since_last_activity'] > min_activity_days
            inactive_materials = material_metrics[inactive_mask]['Материал'].tolist()
            
            # 2. Материалы с недостаточной историей
            insufficient_history_mask = (
                ~inactive_mask & 
                (material_metrics['record_count'] < 5)
            )
            insufficient_history_materials = material_metrics[insufficient_history_mask]['Материал'].tolist()
            
            # 3. Материалы с постоянной ценой
            constant_price_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                (material_metrics['volatility'] < 1)
            )
            constant_price_materials = material_metrics[constant_price_mask]['Материал'].tolist()
            
            # 4. Материалы с высокой волатильностью
            high_volatility_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                (material_metrics['volatility'] > max_volatility)
            )
            high_volatility_materials = material_metrics[high_volatility_mask]['Материал'].tolist()
            
            # 5. Определяем материалы для ML-прогнозирования
            ml_forecasting_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                ~high_volatility_mask & 
                (material_metrics['record_count'] >= min_data_points) & 
                (material_metrics['time_range'] >= 30)
            )
            ml_forecasting_materials = material_metrics[ml_forecasting_mask]['Материал'].tolist()
            
            # 6. Определяем материалы для наивных методов
            naive_forecasting_mask = (
                ~inactive_mask & 
                ~insufficient_history_mask & 
                ~constant_price_mask & 
                ~ml_forecasting_mask & 
                (material_metrics['record_count'] >= 5)
            )
            naive_forecasting_materials = material_metrics[naive_forecasting_mask]['Материал'].tolist()
            
            # Добавляем материалы с высокой волатильностью к наивным методам, если достаточно данных
            naive_high_volatility_mask = (
                high_volatility_mask & 
                (material_metrics['record_count'] >= 5)
            )
            naive_forecasting_materials.extend(
                material_metrics[naive_high_volatility_mask]['Материал'].tolist()
            )
            
            progress_bar.progress(0.7)
            status_text.text("Создаем DataFrame для каждого сегмента...")
            
            # Шаг 3: Создаем DataFrame для каждого сегмента
            # Эффективный способ создания сегментов - фильтрация по спискам материалов
            # Создаем словарь для маппинга материалов в сегменты
            material_to_segment = {}
            
            for material in ml_forecasting_materials:
                material_to_segment[material] = 'ML-прогнозирование'
                
            for material in naive_forecasting_materials:
                material_to_segment[material] = 'Наивные методы'
                
            for material in constant_price_materials:
                material_to_segment[material] = 'Постоянная цена'
                
            for material in inactive_materials:
                material_to_segment[material] = 'Неактивные'
                
            for material in insufficient_history_materials:
                material_to_segment[material] = 'Недостаточно истории'
                
            for material in high_volatility_materials:
                if material not in naive_forecasting_materials:
                    material_to_segment[material] = 'Высокая волатильность'
            
            # Получаем уникальный набор материалов с метриками
            unique_materials_data = material_metrics.copy()
            
            # Добавляем колонку сегмента
            unique_materials_data['Сегмент'] = unique_materials_data['Материал'].map(
                lambda x: material_to_segment.get(x, 'Не классифицирован')
            )
            
            progress_bar.progress(0.9)
            status_text.text("Подготавливаем результаты...")
            
            # Создаем словари для каждого сегмента
            segments = {
                'ML-прогнозирование': unique_materials_data[unique_materials_data['Сегмент'] == 'ML-прогнозирование'],
                'Наивные методы': unique_materials_data[unique_materials_data['Сегмент'] == 'Наивные методы'],
                'Постоянная цена': unique_materials_data[unique_materials_data['Сегмент'] == 'Постоянная цена'],
                'Неактивные': unique_materials_data[unique_materials_data['Сегмент'] == 'Неактивные'],
                'Недостаточно истории': unique_materials_data[unique_materials_data['Сегмент'] == 'Недостаточно истории'],
                'Высокая волатильность': unique_materials_data[unique_materials_data['Сегмент'] == 'Высокая волатильность']
            }
            
            # Переименовываем колонки для совместимости с остальным кодом
            column_mapping = {
                'record_count': 'Количество записей материала',
                'mean_price': 'Средняя цена материала',
                'std_price': 'Стд. отклонение цены материала',
                'volatility': 'Коэффициент вариации цены',
                'is_stable': 'Стабильная цена',
                'time_range': 'Временной диапазон материала',
                'days_since_last_activity': 'Дней с последней активности',
                'last_date': 'Последняя активность материала',
            }
            
            for segment_name, segment_df in segments.items():
                if not segment_df.empty:
                    segments[segment_name] = segment_df.rename(columns=column_mapping)
            
            # Создаем статистику по сегментам
            stats = {
                'ML-прогнозирование': len(ml_forecasting_materials),
                'Наивные методы': len(naive_forecasting_materials),
                'Постоянная цена': len(constant_price_materials),
                'Неактивные': len(inactive_materials),
                'Недостаточно истории': len(insufficient_history_materials),
                'Высокая волатильность': len([m for m in high_volatility_materials if m not in naive_forecasting_materials])
            }
            
            progress_bar.progress(1.0)
            status_text.text("Сегментация завершена!")
            
            # Выводим статистику
            st.write("### Статистика по сегментам")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ML-прогнозирование", stats['ML-прогнозирование'])
                st.metric("Наивные методы", stats['Наивные методы'])
            
            with col2:
                st.metric("Постоянная цена", stats['Постоянная цена'])
                st.metric("Высокая волатильность", stats['Высокая волатильность'])
            
            with col3:
                st.metric("Неактивные", stats['Неактивные'])
                st.metric("Недостаточно истории", stats['Недостаточно истории'])
            
            # Общее количество материалов после сегментации
            total_materials = sum(stats.values())
            
            # Проверка на пересечение сегментов
            if total_materials != num_materials:
                st.warning(f"Внимание: общее количество материалов в сегментах ({total_materials}) "
                           f"не совпадает с количеством уникальных материалов ({num_materials})")
            
            return segments, stats
            
        except Exception as e:
            # В случае ошибки выводим информацию и возвращаем пустые словари
            st.error(f"Ошибка при сегментации материалов: {str(e)}")
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
                "Сегмент материалов:",
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
            st.write(f"Выбран сегмент '{segment_to_export}' с {len(segment_data)} материалами.")
            
            # Опция включения подробной информации
            include_details = st.checkbox("Включить подробную информацию", value=True,
                                        help="Включает дополнительные статистики и метрики для каждого материала")
            
            # Опция экспорта только ключевых колонок
            export_key_columns_only = st.checkbox("Экспортировать только ключевые колонки", value=False,
                                                help="Экспортирует только колонки, необходимые для прогнозирования")
            
            # Кнопка для экспорта
            if st.button("Экспортировать данные"):
                with st.spinner(f"Подготовка данных сегмента '{segment_to_export}'..."):
                    # Получаем полные данные для выбранных материалов
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
            st.warning(f"Сегмент '{segment_to_export}' не содержит материалов.")
    
    def _export_all_segments(self, segments):
        """
        Экспортирует все сегменты в один ZIP-архив
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
        
        # Ограничение количества материалов и строк данных
        max_materials = st.number_input("Максимальное количество материалов на сегмент", min_value=100, value=2000, step=100)
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
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        # Для каждого сегмента
                        segment_count = 0
                        total_segments = len(sorted_segments)
                        
                        for segment_idx, (segment_name, segment_data) in enumerate(sorted_segments):
                            # Обновляем прогресс
                            segment_progress = segment_idx / total_segments
                            overall_progress.progress(segment_progress)
                            status_text.text(f"Обработка сегмента {segment_name} ({segment_idx+1}/{total_segments})...")
                            
                            # Ограничиваем количество материалов для предотвращения переполнения памяти
                            limited_segment_data = segment_data
                            if len(segment_data) > max_materials:
                                limited_segment_data = segment_data.head(max_materials)
                                status_text.text(f"Ограничиваем сегмент {segment_name} до {max_materials} материалов...")
                            
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
        Настраиваемый экспорт с возможностью фильтрации
        """
        st.write("Настраиваемый экспорт с фильтрацией материалов:")
        
        # Опции фильтрации
        st.subheader("Параметры фильтрации")
        
        # 1. Выбор сегментов для экспорта
        selected_segments = st.multiselect(
            "Выберите сегменты для экспорта:",
            list(segments.keys()),
            default=list(segments.keys())
        )
        
        # 2. Фильтры по характеристикам материалов
        with st.expander("Дополнительные фильтры", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                min_records = st.number_input("Минимальное количество записей", min_value=1, value=5)
                max_volatility = st.slider("Максимальный коэффициент вариации (%)", 0, 100, 50)
            
            with col2:
                min_time_range = st.number_input("Минимальный временной диапазон (дни)", min_value=1, value=30)
                max_days_inactive = st.number_input("Максимальные дни неактивности", min_value=0, value=365)
            
            # Ограничение количества материалов и строк
            col1, col2 = st.columns(2)
            with col1:
                max_materials = st.number_input("Макс. количество материалов", min_value=10, value=1000, step=10)
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
                
                # Подсчитываем общее количество материалов до фильтрации
                total_materials_before = sum(len(segments[name]) for name in selected_segments if name in segments)
                status_text.text(f"Анализ {total_materials_before} материалов из {len(selected_segments)} сегментов...")
                
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
                        (segment_data['Количество записей материала'] >= min_records) &
                        (segment_data.get('Коэффициент вариации цены', 0) <= max_volatility) &
                        (segment_data.get('Временной диапазон материала', 0) >= min_time_range) &
                        (segment_data.get('Дней с последней активности', 0) <= max_days_inactive)
                    ]
                    
                    # Добавляем информацию о сегменте
                    if not filtered_segment.empty:
                        filtered_segment['Сегмент'] = segment_name
                        all_filtered_data = pd.concat([all_filtered_data, filtered_segment])
                    
                    # Увеличиваем счетчик обработанных материалов
                    materials_processed += len(segment_data)
                    
                # Ограничиваем количество материалов
                if len(all_filtered_data) > max_materials:
                    status_text.text(f"Ограничение количества материалов до {max_materials}...")
                    all_filtered_data = all_filtered_data.head(max_materials)
                
                progress_bar.progress(0.6)
                
                # Если после фильтрации остались данные
                if not all_filtered_data.empty:
                    status_text.text(f"Подготовка данных для экспорта ({len(all_filtered_data)} материалов)...")
                    
                    # Получаем полные данные для отфильтрованных материалов
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
                        # Создаем маппинг материал -> сегмент
                        material_to_segment = dict(zip(
                            all_filtered_data['Материал'], 
                            all_filtered_data['Сегмент']
                        ))
                        
                        # Добавляем колонку сегмента
                        full_data['Сегмент'] = full_data['Материал'].map(material_to_segment)
                    
                    # Сообщаем о размере данных
                    data_size_mb = full_data.memory_usage(deep=True).sum() / (1024 * 1024)
                    if data_size_mb > 100:
                        st.warning(f"Размер данных: {data_size_mb:.1f} МБ. Экспорт может занять некоторое время.")
                    
                    try:
                        # Экспортируем данные
                        progress_bar.progress(0.9)
                        status_text.text(f"Экспорт данных в формате {export_format}...")
                        
                        if export_format == "CSV":
                            csv_data = self._export_to_csv(full_data)
                            file_name = f"custom_filtered_data.csv"
                            
                            st.download_button(
                                label=f"Скачать {file_name}",
                                data=csv_data,
                                file_name=file_name,
                                mime="text/csv; charset=utf-8-sig"  # Указываем кодировку явно
                            )
                        else:  # Excel
                            excel_data = self._export_to_excel(full_data, "Отфильтрованные данные")
                            file_name = f"custom_filtered_data.xlsx"
                            
                            st.download_button(
                                label=f"Скачать {file_name}",
                                data=excel_data,
                                file_name=file_name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
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
                             f"Количество материалов после фильтрации: {all_filtered_data['Материал'].nunique()}, "
                             f"всего записей: {len(full_data)}")
                    
                    # Показываем сводку по сегментам после фильтрации
                    if 'Сегмент' in all_filtered_data.columns:
                        segment_counts = all_filtered_data.groupby('Сегмент')['Материал'].nunique().reset_index()
                        segment_counts.columns = ['Сегмент', 'Количество материалов']
                        
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
        Получает полные данные для материалов в сегменте
        
        Args:
            segment_data: DataFrame с информацией о материалах в сегменте
            include_details: включать ли детальную информацию о материалах
            key_columns_only: включать только ключевые колонки
            max_rows: максимальное количество строк для экспорта
            
        Returns:
            DataFrame: полные данные для материалов в сегменте
        """
        # Показываем предупреждение о больших объемах данных
        if len(segment_data) > 1000:
            st.warning(f"Выбрано много материалов ({len(segment_data)}). Данные могут обрабатываться дольше обычного.")
        
        # Прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Подготовка данных для экспорта...")
        
        # Получаем список материалов
        materials = segment_data['Материал'].tolist()
        
        # Ограничиваем количество материалов для предотвращения слишком длительной обработки
        if len(materials) > 5000:
            status_text.text(f"Слишком много материалов ({len(materials)}). Ограничиваем до 5000...")
            materials = materials[:5000]
        
        # Получаем полные данные из session_state
        if 'processed_data' in st.session_state:
            full_data = st.session_state.processed_data
            progress_bar.progress(0.1)
            
            # Создаем индекс для быстрого поиска
            status_text.text("Индексирование данных...")
            
            # Вместо применения .isin() ко всему датафрейму, создаем словарь для быстрой проверки
            materials_set = set(materials)
            
            # Используем более эффективный метод фильтрации для больших данных
            if len(full_data) > 100000:
                # Для очень больших данных используем построчную фильтрацию с индикатором прогресса
                filtered_rows = []
                chunk_size = min(len(full_data) // 20, 50000)  # Разбиваем на 20 частей, но не больше 50к строк
                
                for i in range(0, len(full_data), chunk_size):
                    chunk = full_data.iloc[i:i+chunk_size]
                    mask = chunk['Материал'].apply(lambda x: x in materials_set)
                    filtered_chunk = chunk[mask]
                    filtered_rows.append(filtered_chunk)
                    
                    # Обновляем прогресс
                    progress = 0.1 + 0.4 * (i + chunk_size) / len(full_data)
                    progress_bar.progress(min(progress, 0.5))
                    status_text.text(f"Фильтрация данных... {min((i + chunk_size) / len(full_data) * 100, 100):.1f}%")
                
                filtered_data = pd.concat(filtered_rows, ignore_index=True)
            else:
                # Для меньших данных используем стандартный метод
                filtered_data = full_data[full_data['Материал'].isin(materials)].copy()
                progress_bar.progress(0.5)
                status_text.text("Данные отфильтрованы...")
            
            # Ограничиваем количество строк для предотвращения переполнения памяти
            if len(filtered_data) > max_rows:
                status_text.text(f"Слишком много строк данных ({len(filtered_data)}). Ограничиваем до {max_rows}...")
                filtered_data = filtered_data.head(max_rows)
            
            # Если нужны только ключевые колонки
            if key_columns_only:
                key_columns = [
                    'Материал', 'ДатаСоздан', 'Цена нетто', 'Влт', 'Курс', 
                    'Цена нетто (норм.)', 'Год', 'Месяц', 'День'
                ]
                # Оставляем только колонки, которые есть в датафрейме
                existing_key_columns = [col for col in key_columns if col in filtered_data.columns]
                filtered_data = filtered_data[existing_key_columns]
                
                progress_bar.progress(0.7)
                status_text.text("Выбраны ключевые колонки...")
            
            # Если нужно включить детальную информацию о материалах
            if include_details and not key_columns_only:
                status_text.text("Добавление детальной информации...")
                progress_bar.progress(0.6)
                
                # Для каждого материала добавляем его характеристики из segment_data
                # Создаем словари для быстрого доступа к данным
                detail_columns = [
                    'Количество записей материала', 'Коэффициент вариации цены', 
                    'Стабильная цена', 'Временной диапазон материала', 
                    'Дней с последней активности', 'Неактивный материал',
                    'Средняя цена материала', 'Стд. отклонение цены материала'
                ]
                
                # Колонки, которые есть в segment_data
                available_detail_columns = [col for col in detail_columns if col in segment_data.columns]
                
                if available_detail_columns:
                    # Вместо обработки каждой колонки отдельно, создаем промежуточный датафрейм
                    # и используем merge, что должно быть быстрее для больших данных
                    details_df = segment_data[['Материал'] + available_detail_columns].drop_duplicates('Материал')
                    filtered_data = pd.merge(filtered_data, details_df, on='Материал', how='left')
                
                progress_bar.progress(0.8)
                status_text.text("Детальная информация добавлена...")
            
            progress_bar.progress(1.0)
            status_text.text("Данные готовы для экспорта!")
            
            return filtered_data
        
        return pd.DataFrame()  # Возвращаем пустой DataFrame, если нет данных
    
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
            material: код материала для прогнозирования
            forecast_horizon: горизонт прогнозирования в месяцах
        
        Returns:
            dict: подготовленные данные для прогнозирования
        """
        # Получаем данные для указанного материала
        material_data = data[data['Материал'] == material].copy()
        
        # Проверяем, есть ли данные
        if material_data.empty:
            return None
        
        # Сортируем по дате
        material_data = material_data.sort_values('ДатаСоздан')
        
        # Вычисляем частоту данных
        date_diffs = material_data['ДатаСоздан'].diff().dropna()
        
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
        material_data.set_index('ДатаСоздан', inplace=True)
        
        # Ресамплинг данных
        resampled_data = material_data['Цена нетто'].resample(freq).mean()
        
        # Заполняем пропущенные значения
        resampled_data = resampled_data.interpolate(method='linear')
        
        # Определяем последнюю дату в данных
        last_date = resampled_data.index[-1]
        
        # Создаем датаframe для прогнозирования
        forecast_data = {
            'material': material,
            'time_series': resampled_data,
            'freq': freq,
            'last_date': last_date,
            'forecast_horizon': forecast_horizon
        }
        
        return forecast_data