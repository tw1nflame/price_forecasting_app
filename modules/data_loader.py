import streamlit as st
import pandas as pd
import io
import re
import chardet
import copy
import logging
import os # Добавлено для работы с путями

# Настройка логирования
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Класс для загрузки данных из CSV-файла или использования демо-данных
    """
    
    def __init__(self):
        self.encoding_options = ["cp1251", "utf-8", "latin-1", "iso-8859-1", "windows-1251", "koi8-r", "mac-cyrillic"]
        self.delimiter_options = [";", ",", "\t", "|"]
        # Определяем обязательные и рекомендуемые колонки
        self.mandatory_columns = ["Материал", "ДатаСоздан", "Цена нетто"]
        self.recommended_columns = ["за", "ЕЦЗ", "Влт", "Курс", "З-д", "ДокумЗакуп", "ГрЗ", "ГруппаМтр"]
        # Путь к файлу демо-данных
        self.demo_data_path = os.path.join(os.path.dirname(__file__), 'demo_data.csv')
    
    def render(self):
        """
        Отображает интерфейс для загрузки данных
        """
        st.header("Загрузка данных")
        
        st.markdown("""
        Загрузите CSV-файл с данными о материалах или используйте демонстрационный набор данных.
        """)
        
        # Разделяем на две колонки для лучшего размещения
        col1, col2 = st.columns([3, 1]) 
        
        with col1:
            uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"], label_visibility="collapsed")
        
        with col2:
             # Кнопка для загрузки демо-данных
            if st.button("Загрузить демо данные", use_container_width=True):
                self._handle_demo_data_load()
                # Прерываем выполнение render после нажатия кнопки демо-данных, 
                # чтобы не обрабатывать uploaded_file, если он был загружен ранее
                return 

        if uploaded_file is not None:
            self._handle_file_upload(uploaded_file)
            
        # Отображаем информацию о загруженных данных, если они есть
        if 'data' in st.session_state:
            self._display_loaded_data_info()
            
    def _display_loaded_data_info(self):
        """Отображает постоянную информацию о загруженных данных"""
        data = st.session_state.data
        
        st.success(f"✅ Данные загружены: {data.shape[0]} строк, {data.shape[1]} столбцов")
        
    def _handle_file_upload(self, uploaded_file):
        """Обрабатывает загрузку из файла"""
        try:
            # Сразу читаем содержимое файла в память
            file_content = uploaded_file.read()
            file_buffer = io.BytesIO(file_content)
            
            # Опции парсинга CSV
            with st.expander("Настройки импорта"):
                col1_exp, col2_exp = st.columns(2)
                
                with col1_exp:
                    encoding_options = ["auto"] + self.encoding_options
                    encoding = st.selectbox("Кодировка", encoding_options, 
                                          help="Выберите 'auto' для автоматического определения кодировки", key="file_encoding")
                    delimiter = st.selectbox("Разделитель", self.delimiter_options, key="file_delimiter")
                
                with col2_exp:
                    skip_rows = st.number_input("Пропустить строк в начале", min_value=0, value=0, key="file_skip_rows")
                    decimal_separator = st.selectbox("Десятичный разделитель", [".", ","], key="file_decimal_separator")
            
            # Автоопределение кодировки
            if encoding == "auto":
                detection_buffer = io.BytesIO(file_content)
                encoding = self._detect_encoding(detection_buffer)
                st.info(f"Автоматически определена кодировка: {encoding}")
            
            # Предварительный просмотр файла
            try:
                preview_buffer = io.BytesIO(file_content)
                preview_data = self._read_preview(preview_buffer, encoding, delimiter, skip_rows)
                
                st.subheader("Предварительный просмотр")
                from modules.utils import format_streamlit_dataframe
                st.dataframe(
                    format_streamlit_dataframe(preview_data),
                    use_container_width=True,
                    height=400
                )
                
                if st.button("Загрузить данные", key="confirm_file_upload"):
                    with st.spinner("Загрузка данных..."):
                        load_buffer = io.BytesIO(file_content)
                        data = self._load_data(load_buffer, encoding, delimiter, skip_rows, decimal_separator)
                        self._finalize_data_load(data)
                
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {str(e)}")
                st.info("Попробуйте изменить настройки импорта (кодировку, разделитель и т.д.)")
        
        except Exception as e:
            st.error(f"Ошибка при работе с файлом: {str(e)}")
            st.info("Возможно, файл поврежден или имеет неподдерживаемый формат.")

    def _handle_demo_data_load(self):
        """Обрабатывает загрузку демо-данных"""
        if not os.path.exists(self.demo_data_path):
            st.error(f"Ошибка: Файл демо-данных не найден по пути: {self.demo_data_path}")
            return
            
        try:
            with st.spinner("Загрузка демо данных..."):
                with open(self.demo_data_path, 'rb') as f:
                    demo_content = f.read()
                
                demo_buffer = io.BytesIO(demo_content)
                
                # Используем фиксированные параметры для демо-данных
                encoding = 'utf-8' # Демо файл сохранен в utf-8
                delimiter = ';'
                skip_rows = 0
                decimal_separator = ',' # В демо файле используется запятая

                # Убедимся, что кодировка utf-8 есть в списке для _load_data
                if encoding not in self.encoding_options:
                    self.encoding_options.append(encoding) 
                
                data = self._load_data(demo_buffer, encoding, delimiter, skip_rows, decimal_separator)
                self._finalize_data_load(data)

        except Exception as e:
            st.error(f"Ошибка при загрузке демо-данных: {str(e)}")
            # Логируем ошибку для детальной диагностики
            logger.error(f"Error loading demo data from {self.demo_data_path}: {e}", exc_info=True)
            if 'data' in st.session_state: del st.session_state.data # Очистка на случай частичной загрузки

    def _finalize_data_load(self, data):
        """Общая логика после загрузки данных (из файла или демо)"""
        if data is not None:
            # Проверка на наличие обязательных колонок — теперь не блокирует загрузку.
            missing_mandatory = [col for col in self.mandatory_columns if col not in data.columns]
            if missing_mandatory:
                st.warning(
                    f"Внимание: В загруженном файле не найдены обязательные колонки: {', '.join(missing_mandatory)}. "
                    "Данные загружены — выполните маппинг колонок в разделе 'Маппинг колонок'."
                )

            # Проверка на наличие рекомендуемых колонок
            missing_recommended = [col for col in self.recommended_columns if col not in data.columns]
            if missing_recommended:
                st.warning(f"Внимание: Отсутствуют рекомендуемые колонки: {', '.join(missing_recommended)}. Некоторые функции анализа могут быть недоступны.")

            # Сохраняем данные и оригинальные имена колонок в session_state
            st.session_state.data = data
            st.session_state.original_columns = list(data.columns)
            
            # Очищаем другие сессионные данные
            keys_to_clear = ['processed_data', 'materials_segments', 'segments_stats', 'stability_data', 'volatility_data', 'security_risks'] # Добавлены все возможные ключи
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
        else:
            # Если _load_data вернул None, значит была ошибка при загрузке/парсинге
            st.error("Не удалось загрузить данные. Проверьте логи или настройки импорта.")
            if 'data' in st.session_state: del st.session_state.data # Убедимся, что данные очищены

    def _detect_encoding(self, file_buffer):
        """
        Определяет кодировку файла
        
        Args:
            file_buffer: io.BytesIO с содержимым файла
        
        Returns:
            str: определенная кодировка
        """
        try:
            # Читаем часть файла для определения кодировки
            sample = file_buffer.read(min(10000, file_buffer.getbuffer().nbytes))
            
            # Используем chardet для определения кодировки
            detection = chardet.detect(sample)
            detected_encoding = detection['encoding']
            
            # Если кодировка определена и имеет высокую вероятность, используем её
            if detected_encoding and detection['confidence'] > 0.7:
                return detected_encoding
            
            # Иначе возвращаем cp1251 (наиболее вероятная для русскоязычных данных)
            return "cp1251"
        except Exception as e:
            st.warning(f"Ошибка при определении кодировки: {str(e)}. Используется cp1251.")
            return "cp1251"
    
    def _read_preview(self, file_buffer, encoding, delimiter, skip_rows):
        """
        Читает первые 5 строк из файла для предварительного просмотра
        
        Args:
            file_buffer: io.BytesIO с содержимым файла
            encoding: кодировка файла
            delimiter: разделитель
            skip_rows: количество строк для пропуска
        
        Returns:
            pandas.DataFrame: предварительный просмотр данных
        """
        # Пробуем различные кодировки, если указанная не работает
        encodings_to_try = [encoding] + [enc for enc in self.encoding_options if enc != encoding]
        
        for enc in encodings_to_try:
            try:
                # Сбрасываем буфер в начало
                file_buffer.seek(0)
                
                # Декодируем содержимое с текущей кодировкой
                text_content = file_buffer.read().decode(enc)
                lines = text_content.splitlines()
                
                # Пропускаем строки, если необходимо
                if skip_rows > 0:
                    lines = lines[skip_rows:]
                
                # Берем первые 5 строк для предварительного просмотра
                preview_lines = lines[:5]
                
                # Объединяем строки
                preview_text = "\n".join(preview_lines)
                preview_string_buffer = io.StringIO(preview_text)
                
                try:
                    # Пытаемся разобрать как CSV
                    preview_df = pd.read_csv(preview_string_buffer, delimiter=delimiter, encoding="utf-8")
                    
                    # Если нам удалось разобрать файл с другой кодировкой, сообщаем об этом
                    if enc != encoding and encoding != "auto":
                        st.info(f"Для предварительного просмотра используется кодировка {enc} вместо {encoding}")
                    
                    return preview_df
                
                except Exception:
                    # Если не удалось разобрать как CSV, пробуем ручной разбор
                    preview_rows = []
                    for line in preview_lines:
                        row = line.strip().split(delimiter)
                        preview_rows.append(row)
                    
                    # Определяем максимальное количество столбцов
                    max_cols = max((len(row) for row in preview_rows), default=0)
                    
                    # Создаем заголовки, если их нет
                    if preview_rows:
                        if len(preview_rows[0]) == max_cols:
                            headers = preview_rows[0]
                            data_rows = preview_rows[1:]
                        else:
                            headers = [f"Column {i+1}" for i in range(max_cols)]
                            data_rows = preview_rows
                        
                        # Обеспечиваем, чтобы все строки имели одинаковое количество столбцов
                        for i in range(len(data_rows)):
                            if len(data_rows[i]) < max_cols:
                                data_rows[i].extend([""] * (max_cols - len(data_rows[i])))
                        
                        # Создаем DataFrame
                        preview_df = pd.DataFrame(data_rows, columns=headers)
                        
                        # Если нам удалось разобрать файл с другой кодировкой, сообщаем об этом
                        if enc != encoding and encoding != "auto":
                            st.info(f"Для предварительного просмотра используется кодировка {enc} вместо {encoding}")
                        
                        return preview_df
            
            except UnicodeDecodeError:
                # Пробуем следующую кодировку
                continue
            except Exception as e:
                st.warning(f"Ошибка при чтении файла с кодировкой {enc}: {str(e)}")
                continue
        
        # Если не удалось прочитать с любой кодировкой
        raise Exception("Не удалось прочитать файл с указанными параметрами. Попробуйте изменить кодировку или разделитель.")
    
    def _load_data(self, file_buffer, encoding, delimiter, skip_rows, decimal_separator):
        """
        Загружает данные из файла в pandas DataFrame
        
        Args:
            file_buffer: io.BytesIO с содержимым файла
            encoding: кодировка файла
            delimiter: разделитель
            skip_rows: количество строк для пропуска
            decimal_separator: десятичный разделитель
        
        Returns:
            pandas.DataFrame: загруженные данные
        """
        try:
            # Пробуем различные кодировки, если указанная не работает
            encodings_to_try = [encoding] + [enc for enc in self.encoding_options if enc != encoding]
            
            # Хранение ошибок для диагностики
            errors = []
            
            for enc in encodings_to_try:
                try:
                    # Сбрасываем буфер в начало
                    file_buffer.seek(0)
                    
                    # Декодируем содержимое файла
                    text_content = file_buffer.read().decode(enc)
                    
                    # Если десятичный разделитель - запятая, заменяем его на точку
                    if decimal_separator == ",":
                        # Заменяем разделители в числах, сохраняя разделители в датах
                        def replace_decimal_separator(match):
                            # Заменяем запятые в числах на точки
                            value = match.group(0)
                            if "," in value and "." not in value:
                                return value.replace(",", ".")
                            return value
                        
                        # Ищем числа с разделителями
                        text_content = re.sub(r'\b\d+[,]\d+\b', replace_decimal_separator, text_content)
                    
                    # Пропускаем строки, если необходимо
                    if skip_rows > 0:
                        lines = text_content.splitlines()
                        text_content = "\n".join(lines[skip_rows:])
                    
                    # Создаем StringIO для pandas
                    string_buffer = io.StringIO(text_content)
                    
                    # Читаем данные
                    data = pd.read_csv(string_buffer, delimiter=delimiter)
                    
                    # Если доходим до сюда, значит кодировка подошла
                    if enc != encoding and encoding != "auto":
                        st.info(f"Для загрузки данных используется кодировка {enc} вместо {encoding}")
                    
                    # Ранее код автоматически переименовывал колонки в ожидаемые
                    # имена при совпадении количества столбцов. Это приводило к
                    # потере оригинальных заголовков файла и мешало ручному маппингу.
                    # Теперь возвращаем DataFrame с оригинальными заголовками.
                    return data
                
                except Exception as e:
                    errors.append(f"Ошибка при использовании кодировки {enc}: {str(e)}")
                    continue
            
            # Если все кодировки не подошли, выводим все ошибки
            error_message = "\n".join(errors)
            # Логируем подробную ошибку
            logger.error(f"Failed to load data with any encoding: {error_message}")
            raise Exception(f"Не удалось загрузить данные ни с одной кодировкой. Проверьте файл и настройки импорта.")
        
        except Exception as e:
            # Логируем исключение
            logger.error(f"Error during data loading: {e}", exc_info=True)
            from modules.utils import show_error_message
            show_error_message(e, "Ошибка при загрузке данных", show_traceback=False) # Отключаем traceback для пользователя
            st.info("Попробуйте изменить параметры загрузки или используйте другой файл.")
            return None # Возвращаем None в случае ошибки