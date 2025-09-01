import streamlit as st
import pandas as pd
import os
import sys

# Увеличиваем лимит элементов для отображения в Pandas Styler
pd.set_option("styler.render.max_elements", 1000000)  # Устанавливаем лимит в 1 миллион элементов

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модули
from modules.data_loader import DataLoader
from modules.data_processor import DataProcessor
from modules.data_analyzer import DataAnalyzer
from modules.visualization import Visualizer
from modules.material_segmentation import MaterialSegmenter
from modules.forecasting_preparation import ForecastPreparation
from modules.security_analyzer import SecurityAnalyzer
from modules.utils import apply_custom_css, show_user_guide, show_performance_info, show_app_version

# Настройка страницы
st.set_page_config(
    page_title="Анализ и прогнозирование временных рядов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Применяем пользовательский CSS
apply_custom_css()

# Создаем объекты модулей
data_loader = DataLoader()
# Centralized role names used across modules (canonical role identifiers)
ROLE_NAMES = {
    'ROLE_ID': 'ID',
    'ROLE_DATE': 'Дата',
    'ROLE_TARGET': 'Целевая Колонка',
    'ROLE_QTY': 'Количество',
    'ROLE_CURRENCY': 'Валюта',
    'ROLE_RATE': 'Курс'
}

data_processor = DataProcessor(ROLE_NAMES)
data_analyzer = DataAnalyzer(ROLE_NAMES)
visualizer = Visualizer(ROLE_NAMES)
material_segmenter = MaterialSegmenter(ROLE_NAMES)
forecast_preparation = ForecastPreparation(ROLE_NAMES)
security_analyzer = SecurityAnalyzer()
security_analyzer.set_role_names(ROLE_NAMES)

# Боковая панель
with st.sidebar:
    st.header("Навигация по временным рядам")
    page = st.radio(
        "Выберите раздел:",
        ["Информация", "Загрузка данных", "Общий анализ", "Анализ уникальности временных рядов",
         "Временной анализ", "Анализ волатильности", "Стабильные временные ряды",
         "Неактивные временные ряды", "Сегментация временных рядов для прогнозирования", "Анализ безопасности", "Экспорт данных"]
    )
    st.divider()
    st.header("Статус")
    # Проверка наличия данных в сессии
    if 'data' in st.session_state:
        st.success(f"Данные временных рядов загружены: {st.session_state.data.shape[0]} строк")
        if 'processed_data' in st.session_state:
            st.success("Данные временных рядов обработаны")
        if 'materials_segments' in st.session_state:
            st.success("Временные ряды сегментированы")
            # Отображение статистики по сегментам
            segments_stats = st.session_state.get('segments_stats', {})
            if segments_stats:
                st.write("Распределение временных рядов по сегментам:")
                for segment, count in segments_stats.items():
                    st.write(f"- {segment}: {count}")
    else:
        st.warning("Данные временных рядов не загружены")

# Заголовок приложения
st.title("Анализ и прогнозирование временных рядов и их целевых колонок")

# Функция для отображения информации
def display_info():
     st.info("""
     Это приложение предназначено для анализа временных рядов и их целевых колонок,
     а также для подготовки к прогнозированию значений целевых колонок с помощью различных методов машинного обучения.
    
     Загрузите CSV-файл с данными временных рядов, и приложение выполнит анализ,
     определит временные ряды, подходящие для прогнозирования, и подготовит данные для дальнейшего использования.
     """)
     # Добавляем руководство пользователя
     with st.expander("Руководство пользователя по работе с временными рядами", expanded=False):
          show_user_guide()
     # Добавляем информацию для служб безопасности
     with st.expander("Информация для служб безопасности"):
          st.markdown("""
          ## Руководство для служб информационной безопасности
        
          Данное приложение включает специальный модуль для выявления подозрительных паттернов во временных рядах, которые могут указывать на:
          - Потенциальные мошеннические схемы
          - Манипуляции с целевыми колонками
          - Дробление событий для обхода процедур
          - Концентрацию активности в конце отчетных периодов
          - Нехарактерные изменения во временных рядах
        
          ### Ключевые индикаторы риска:
        
          1. **Высокая волатильность целевой колонки**
              *   **Что это:** Значения целевой колонки одного и того же временного ряда сильно колеблются без видимых причин.
              *   **Как измеряется:** Рассчитывается коэффициент вариации (CV). Высоким считается CV > 50% (значение в среднем отклоняется более чем на половину от среднего).
              *   **Пример:** Значение целевой колонки вчера было 100, сегодня - 250, а через неделю - 80, при этом внешних причин для таких скачков не было. Это может указывать на манипуляции или ошибки.
        
          2. **Дробление событий**
              *   **Что это:** Вместо одного крупного события проводится множество мелких, часто с небольшими интервалами времени. Цель может быть - обойти процедуру согласования или контроля.
              *   **Как измеряется:** Анализируется частота событий одного и того же временного ряда на значения, близкие к пороговым.
              *   **Пример:** В течение недели по одному ряду фиксируется 5 событий с близкими значениями, хотя можно было бы объединить их в одно. Если порог для контроля - 50, это подозрительно.
        
          3. **Активность в конце периодов**
              *   **Что это:** Непропорционально большое количество событий или повышение значений целевой колонки происходит в последние дни отчетного периода (месяц, квартал, год).
              *   **Причина:** Может указывать на попытки срочно "освоить бюджет" или на спешку, ведущую к аномалиям.
              *   **Пример:** 80% всех событий по ряду за квартал приходится на последние два дня марта. Или значение целевой колонки систематически повышается в декабре.
        
          4. **Аномальная стабильность целевой колонки**
              *   **Что это:** Значение целевой колонки остается абсолютно одинаковым в течение длительного времени, даже если по логике оно должно меняться. Также сюда относятся подозрительно "круглые" значения.
              *   **Причина:** Может указывать на фиктивные данные или отсутствие анализа.
              *   **Пример:** Значение целевой колонки всегда ровно 100.00 на протяжении двух лет без изменений.
        
          5. **Нарушения сезонности**
              *   **Что это:** События происходят в периоды, когда их обычно не бывает, либо динамика целевой колонки противоречит общим тенденциям.
              *   **Причина:** Может указывать на неэффективное планирование или нестандартные схемы.
              *   **Пример:** Массовая активность по ряду происходит в июне, хотя обычно она бывает в декабре.
        
          ### Как использовать модуль анализа безопасности:
        
          1. Загрузите и обработайте данные
          2. Выполните сегментацию временных рядов
          3. Перейдите в раздел "Анализ безопасности"
          4. Изучите временные ряды с высоким индексом подозрительности
          5. Выполните детальный анализ временных рядов из группы риска
          6. Экспортируйте отчет для дальнейшего расследования
        
          **Примечание:** Особое внимание следует уделить временным рядам из сегмента "Высокая волатильность", так как в этом сегменте наиболее часто обнаруживаются признаки потенциальных нарушений.
          """)

# Основное содержимое
if page == "Информация":
    display_info()

elif page == "Загрузка данных":
    data_loader.render()

    # Показываем форму для маппинга колонок только если данные загружены
    if 'data' in st.session_state:
        st.header("Картирование колонок (mapping)")

        cols = list(st.session_state.data.columns)

        # Инициализируем значения в session_state, если их нет
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = {}

        # Показываем подсказку
        st.caption("Выберите соответствующие колонки из загруженных данных. Сохранение происходит автоматически.")

        # Используем уникальные ключи для виджетов, чтобы значения сохранялись
        id_choice = st.selectbox("1. Идентификационная колонка", ["(не выбрано)"] + cols, key='map_id')
        date_choice = st.selectbox("2. Дата", ["(не выбрано)"] + cols, key='map_date')
        target_choice = st.selectbox("3. Целевая колонка", ["(не выбрано)"] + cols, key='map_target')
        qty_choice = st.selectbox("4. Количество (за)", ["(не выбрано)"] + cols, key='map_quantity')
        currency_choice = st.selectbox("5. Валюта", ["(не выбрано)"] + cols, key='map_currency')
        rate_choice = st.selectbox("6. Курс", ["(не выбрано)"] + cols, key='map_rate')

        # Собираем маппинг и сохраняем в session_state
        mapping = {
            'ID': None if id_choice == "(не выбрано)" else id_choice,
            'Дата': None if date_choice == "(не выбрано)" else date_choice,
            'Целевая Колонка': None if target_choice == "(не выбрано)" else target_choice,
            'Количество': None if qty_choice == "(не выбрано)" else qty_choice,
            'Валюта': None if currency_choice == "(не выбрано)" else currency_choice,
            'Курс': None if rate_choice == "(не выбрано)" else rate_choice,
        }

        st.session_state.column_mapping = mapping

        # Кнопка обработки данных должна быть активна только если выбраны ID, Date и Target
        required_ok = mapping['ID'] is not None and mapping['Дата'] is not None and mapping['Целевая Колонка'] is not None

        if not required_ok:
            st.info("Для активации кнопки 'Обработать данные' выберите: 1) Идентификационную колонку, 2) Дату, 3) Целевую колонку.")

        # Кнопка с контролем disabled
        if st.button("Обработать данные", disabled=not required_ok):
            with st.spinner("Обработка данных..."):
                mapping = st.session_state.get('column_mapping', None)
                # Передаем mapping, он может содержать None для невыбранных ролей
                st.session_state.processed_data = data_processor.process_data(st.session_state.data, column_mapping=mapping)
                st.success("Данные успешно обработаны!")
                # Показать образец обработанных данных
                st.subheader("Образец обработанных данных")
                st.dataframe(st.session_state.processed_data.head())

elif page == "Общий анализ":
    if 'processed_data' in st.session_state:
        data_analyzer.render_overview(st.session_state.processed_data)
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Анализ уникальности временных рядов":
    if 'processed_data' in st.session_state:
        data_analyzer.render_materials_uniqueness(st.session_state.processed_data)
        visualizer.plot_materials_distribution(st.session_state.processed_data)
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Временной анализ":
    if 'processed_data' in st.session_state:
        data_analyzer.render_time_analysis(st.session_state.processed_data)
        visualizer.plot_time_distribution(st.session_state.processed_data)
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Анализ волатильности":
    if 'processed_data' in st.session_state:
        # Проверяем, есть ли уже данные анализа волатильности
        if 'volatility_data' not in st.session_state:
            with st.spinner("Анализ волатильности..."):
                # Предполагаем, что эта функция сохраняет результат в st.session_state.volatility_data
                material_segmenter.analyze_volatility(st.session_state.processed_data)

        # Отображаем, если данные есть (были или только что рассчитаны)
        if 'volatility_data' in st.session_state:
            visualizer.plot_volatility(st.session_state.volatility_data)
        else:
            st.error("Не удалось выполнить или загрузить данные анализа волатильности.")
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Стабильные временные ряды":
    if 'processed_data' in st.session_state:
        # Проверяем, есть ли уже данные анализа стабильности
        if 'stability_data' not in st.session_state:
            with st.spinner("Анализ стабильности временных рядов..."):
                 # Предполагаем, что эта функция сохраняет результат в st.session_state.stability_data
                material_segmenter.analyze_stability(st.session_state.processed_data)

        # Отображаем, если данные есть
        if 'stability_data' in st.session_state:
            visualizer.plot_stability(st.session_state.stability_data)
        else:
            st.error("Не удалось выполнить или загрузить данные анализа стабильности.")
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Неактивные временные ряды":
    if 'processed_data' in st.session_state:
         # Проверяем, есть ли уже данные анализа неактивности
        if 'inactivity_data' not in st.session_state:
            with st.spinner("Анализ неактивных временных рядов..."):
                 # Предполагаем, что эта функция сохраняет результат в st.session_state.inactivity_data
                material_segmenter.analyze_inactivity(st.session_state.processed_data)

        # Отображаем, если данные есть
        if 'inactivity_data' in st.session_state:
            visualizer.plot_inactivity(st.session_state.inactivity_data)
        else:
             st.error("Не удалось выполнить или загрузить данные анализа неактивности.")
    else:
        st.warning("Сначала загрузите и обработайте данные")

elif page == "Сегментация временных рядов для прогнозирования":
    if 'processed_data' in st.session_state:
        st.header("Сегментация временных рядов для прогнозирования")
        
        # Параметры сегментации
        with st.expander("Параметры сегментации", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_data_points = st.slider("Минимальное количество точек данных", 5, 100, 24)
            
            with col2:
                max_volatility = st.slider("Максимальный коэффициент вариации (%)", 1, 100, 30)
            
            with col3:
                min_activity_days = st.slider("Минимальное количество дней активности", 30, 1000, 365)
        
        if st.button("Выполнить сегментацию"):
            with st.spinner("Сегментация временных рядов..."):
                segments, stats = forecast_preparation.segment_materials(
                    st.session_state.processed_data,
                    min_data_points=min_data_points,
                    max_volatility=max_volatility,
                    min_activity_days=min_activity_days
                )

                st.session_state.materials_segments = segments
                st.session_state.segments_stats = stats

                st.success("Сегментация завершена!")

                # Очищаем состояние виджетов визуализации при новой сегментации
                keys_to_clear = ['vis_seg_details_select', 'vis_seg_page_size', 'vis_seg_page_number']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                # Визуализацию вызываем ниже, после проверки наличия данных в сессии

        # Отображение результатов, если они есть в сессии
        if 'materials_segments' in st.session_state and 'segments_stats' in st.session_state:
             st.subheader("Результаты сегментации")

             # --- Пояснение к графику распределения ---
             st.markdown("""
             #### Распределение временных рядов по сегментам
             График ниже показывает, как временные ряды распределились по сегментам после анализа на основе заданных параметров (`Минимальное количество точек данных`, `Максимальный коэффициент вариации`, `Минимальное количество дней активности`).
             - **Подходит для прогнозирования:** Временные ряды с достаточным количеством исторических данных, относительно стабильными значениями целевой колонки и событиями, происходившими на протяжении достаточного периода. Именно для этих временных рядов имеет смысл строить прогнозы.
             - **Недостаточно данных:** Временные ряды, по которым записей меньше, чем `Минимальное количество точек данных`. Прогнозирование по ним будет ненадежным из-за малого объема выборки.
             - **Нестабильный (Высокая волатильность):** Временные ряды, коэффициент вариации значений целевой колонки которых превышает `Максимальный коэффициент вариации`. Сильные колебания значений затрудняют прогнозирование.
             - **Недостаточно активный:** Временные ряды, период между первым и последним событием по которым меньше, чем `Минимальное количество дней активности`. Короткая история не позволяет выявить тренды.

             *Бизнес-смысл:* Этот график дает общее представление о том, какая доля временных рядов подходит для автоматического прогнозирования. Например, если большая часть временных рядов попадает в сегмент "Нестабильный", это может указывать на необходимость пересмотра соответствующих процессов или анализа причин такой волатильности для ключевых рядов.
             """)
             # --- Конец пояснения к графику ---

             # Используем данные из session_state
             visualizer.plot_segmentation_results(
                 st.session_state.materials_segments,
                 st.session_state.segments_stats
             )

             # --- Пояснение к таблице(ам) детализации ---
             st.markdown("""
             ---
             #### Детализация по сегментам
             В таблице (или таблицах, если выбрано отображение по сегментам) ниже представлен список временных рядов, попавших в каждый сегмент. Для каждого временного ряда указаны рассчитанные метрики: количество точек данных, коэффициент вариации (%) и длительность периода активности (дни).

             *Бизнес-смысл:* Эта таблица позволяет детально изучить состав каждого сегмента. Вы можете отсортировать или отфильтровать таблицу, чтобы найти конкретные временные ряды. Например, можно посмотреть, какие именно ряды попали в сегмент "Подходит для прогнозирования", чтобы сфокусироваться на них, или проанализировать временные ряды из сегмента "Недостаточно данных", чтобы оценить возможность сбора дополнительной информации по ним.
             """)
             # --- Конец пояснения к таблице ---

        # Сообщение, если сегментация еще не проводилась (но данные загружены)
        elif 'materials_segments' not in st.session_state:
             st.info("Задайте параметры и нажмите 'Выполнить сегментацию' для просмотра результатов.")

    else:
        st.warning("Сначала загрузите и обработайте данные")
elif page == "Анализ безопасности":
    if 'processed_data' in st.session_state and 'materials_segments' in st.session_state:

        # Кнопка для повторного запуска анализа
        if st.button("Повторить анализ"):
            if 'security_risks' in st.session_state:
                del st.session_state['security_risks']
            st.rerun()

        # Запускаем анализ ТОЛЬКО если результаты еще не сохранены в сессии
        if 'security_risks' not in st.session_state:
            with st.spinner("Анализ безопасности данных..."):
                st.session_state.security_risks = security_analyzer.analyze_security_risks(
                    st.session_state.processed_data,
                    st.session_state.materials_segments
                )

        # Отображаем результаты анализа, если они есть
        if 'security_risks' in st.session_state and st.session_state.security_risks is not None and not st.session_state.security_risks.empty:
            risk_df = st.session_state.security_risks
            security_analyzer.display_security_analysis_results(risk_df)

            st.divider()
            st.subheader("Детальный анализ временных рядов с высоким риском")

            high_risk_materials_df = risk_df[risk_df['Категория риска'] == 'Высокий']

            if not high_risk_materials_df.empty:
                # use canonical role id column if present, otherwise fallback to first column
                idcol = ROLE_NAMES.get('ROLE_ID') if ROLE_NAMES.get('ROLE_ID') in high_risk_materials_df.columns else high_risk_materials_df.columns[0]
                high_risk_options = high_risk_materials_df[idcol].tolist()
                
                selected_materials = st.multiselect(
                    "Выберите временные ряды для детального анализа:",
                    options=high_risk_options,
                    default=high_risk_options[:min(5, len(high_risk_options))] # По умолчанию выбираем первые 5 или меньше
                )

                if selected_materials:
                    # Генерируем данные для Excel
                    excel_data = security_analyzer.export_multiple_detailed_analysis(
                        st.session_state.processed_data, 
                        selected_materials
                    )
                    
                    if excel_data:
                        st.download_button(
                           label="Скачать детальный анализ для выбранных временных рядов (Excel)",
                           data=excel_data,
                           file_name="detailed_security_analysis_multiple.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.info("Выберите хотя бы один временной ряд для загрузки детального анализа.")
            else:
                st.info("Временные ряды с высоким риском не найдены.")

        elif 'security_risks' in st.session_state and st.session_state.security_risks is not None and st.session_state.security_risks.empty:
             st.info("Анализ безопасности завершен, но временные ряды с признаками риска не обнаружены.")
        else:
             st.info("Результаты анализа безопасности отсутствуют или не были сгенерированы.")

    elif 'processed_data' not in st.session_state:
        st.warning("Сначала загрузите и обработайте данные")
    else:
        st.warning("Сначала выполните сегментацию временных рядов в разделе 'Сегментация временных рядов для прогнозирования'")
        
        if st.button("Перейти к сегментации"):
            st.session_state.page = "Сегментация временных рядов для прогнозирования"
            st.rerun()
    
elif page == "Экспорт данных":
    if 'materials_segments' in st.session_state:
        st.header("Экспорт данных")
        
        st.markdown("""
        ## Экспорт данных для использования в других приложениях
        
        В этом разделе вы можете экспортировать данные различных сегментов временных рядов
        в форматы CSV или Excel для дальнейшего использования в других приложениях.
        
        Доступные опции экспорта:
        - **Экспорт по сегментам** - экспорт данных отдельно для каждого сегмента
        - **Массовый экспорт** - экспорт всех сегментов в один ZIP-архив
        - **Настраиваемый экспорт** - экспорт с возможностью фильтрации по различным параметрам
        """)
        
        # Вызываем метод экспорта данных
        forecast_preparation.export_data_options(st.session_state.materials_segments)
    else:
        st.warning("Сначала выполните сегментацию временных рядов в разделе 'Сегментация временных рядов для прогнозирования'")
        
        if st.button("Перейти к сегментации"):
            # Переключаемся на страницу сегментации
            st.session_state.page = "Сегментация временных рядов для прогнозирования"
            st.rerun()

# Footer
st.divider()

# Отображение информации о производительности
show_performance_info()

# Отображение информации о версии
show_app_version()

st.caption("© 2025 Анализ и прогнозирование временных рядов")