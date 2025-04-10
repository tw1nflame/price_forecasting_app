import streamlit as st
import pandas as pd
import io
import re
import chardet
import copy

class DataLoader:
    """
    Класс для загрузки данных из CSV-файла
    """
    
    def __init__(self):
        self.encoding_options = ["cp1251", "utf-8", "latin-1", "iso-8859-1", "windows-1251", "koi8-r", "mac-cyrillic"]
        self.delimiter_options = [";", ",", "\t", "|"]
    
    def render(self):
        """
        Отображает интерфейс для загрузки данных
        """
        st.header("Загрузка данных")
        
        st.write("""
        Загрузите CSV-файл с данными о материалах. 
        Ожидаемые колонки: Материал, ДатаСоздан, Цена нетто, за, ЕЦЗ, Влт, Курс, З-д, ДокумЗакуп, ГрЗ, ГруппаМтр
        """)
        
        uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Сразу читаем содержимое файла в память, чтобы избежать проблем с закрытым файлом
                file_content = uploaded_file.read()
                file_buffer = io.BytesIO(file_content)
                
                # Опции парсинга CSV
                with st.expander("Настройки импорта"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        encoding_options = ["auto"] + self.encoding_options
                        encoding = st.selectbox("Кодировка", encoding_options, 
                                              help="Выберите 'auto' для автоматического определения кодировки")
                        delimiter = st.selectbox("Разделитель", self.delimiter_options)
                    
                    with col2:
                        skip_rows = st.number_input("Пропустить строк в начале", min_value=0, value=0)
                        decimal_separator = st.selectbox("Десятичный разделитель", [".", ","])
                
                # Автоопределение кодировки, если выбрано "auto"
                if encoding == "auto":
                    # Создаем копию буфера, чтобы не влиять на исходный буфер
                    detection_buffer = io.BytesIO(file_content)
                    encoding = self._detect_encoding(detection_buffer)
                    st.info(f"Автоматически определена кодировка: {encoding}")
                
                # Предварительный просмотр файла
                try:
                    # Создаем новую копию буфера для предварительного просмотра
                    preview_buffer = io.BytesIO(file_content)
                    preview_data = self._read_preview(preview_buffer, encoding, delimiter, skip_rows)
                    
                    st.subheader("Предварительный просмотр")
                    from modules.utils import format_streamlit_dataframe
                    st.dataframe(
                        format_streamlit_dataframe(preview_data),
                        use_container_width=True,
                        height=400  # Фиксированная высота для лучшего отображения
                    )
                    
                    if st.button("Загрузить данные"):
                        with st.spinner("Загрузка данных..."):
                            # Создаем новую копию буфера для полной загрузки
                            load_buffer = io.BytesIO(file_content)
                            
                            # Загружаем весь файл
                            data = self._load_data(load_buffer, encoding, delimiter, skip_rows, decimal_separator)
                            
                            # Сохраняем данные в session_state
                            st.session_state.data = data
                            
                            # Очищаем другие сессионные данные, если они существуют
                            if 'processed_data' in st.session_state:
                                del st.session_state.processed_data
                            if 'materials_segments' in st.session_state:
                                del st.session_state.materials_segments
                            if 'segments_stats' in st.session_state:
                                del st.session_state.segments_stats
                            
                            st.success(f"Данные успешно загружены! Загружено {data.shape[0]} строк и {data.shape[1]} столбцов.")
                            
                            # Показываем информацию о данных
                            st.subheader("Информация о данных")
                            st.write(f"Количество строк: {data.shape[0]}")
                            st.write(f"Количество столбцов: {data.shape[1]}")
                            
                            # Показываем пример данных
                            st.subheader("Пример данных")
                            from modules.utils import format_streamlit_dataframe
                            st.dataframe(
                                format_streamlit_dataframe(data.head()),
                                use_container_width=True,
                                height=400  # Фиксированная высота для лучшего отображения
                            )
                    
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {str(e)}")
                    st.info("Попробуйте изменить настройки импорта (кодировку, разделитель и т.д.)")
            
            except Exception as e:
                st.error(f"Ошибка при работе с файлом: {str(e)}")
                st.info("Возможно, файл поврежден или имеет неподдерживаемый формат.")
    
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
                    
                    # Проверяем, что колонки соответствуют ожидаемым
                    expected_columns = [
                        "Материал", "ДатаСоздан", "Цена нетто", "за", "ЕЦЗ", "Влт", 
                        "Курс", "З-д", "ДокумЗакуп", "ГрЗ", "ГруппаМтр"
                    ]
                    
                    # Если количество столбцов совпадает, но названия разные, 
                    # переименовываем их
                    if len(data.columns) == len(expected_columns) and not all(col in data.columns for col in expected_columns):
                        data.columns = expected_columns
                    
                    return data
                
                except Exception as e:
                    errors.append(f"Ошибка при использовании кодировки {enc}: {str(e)}")
                    continue
            
            # Если все кодировки не подошли, выводим все ошибки
            error_message = "\n".join(errors)
            raise Exception(f"Не удалось загрузить данные ни с одной кодировкой:\n{error_message}")
        
        except Exception as e:
            raise Exception(f"Ошибка при загрузке данных: {str(e)}")