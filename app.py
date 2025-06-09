import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import csr_matrix
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleRecommenderSystem:
    def __init__(self):
        self.svd_model = None
        self.nmf_model = None
        self.rf_model = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.scaler = StandardScaler()
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.processed_data = None
        self.feature_columns = []
        self.weights = {'svd': 0.4, 'nmf': 0.3, 'similarity': 0.2, 'content': 0.1}
        
    def parse_dates(self, df, date_column='Datasales'):
        """
        Улучшенная функция для парсинга дат с множественными форматами
        """
        df = df.copy()
        
        # Список возможных форматов дат
        date_formats = [
            '%Y-%m-%d',
            '%d.%m.%Y',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y.%m.%d',
            '%d.%m.%y',
            '%d/%m/%y',
            '%m/%d/%y',
            '%y/%m/%d',
            '%d-%m-%y',
            '%m-%d-%y'
        ]
        
        if date_column not in df.columns:
            logger.error(f"Колонка {date_column} не найдена в данных")
            raise ValueError(f"Колонка {date_column} не найдена")
        
        # Сначала попробуем pandas.to_datetime с автоматическим определением
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=True)
            parsed_count = df[date_column].notna().sum()
            total_count = len(df)
            
            if parsed_count == total_count:
                logger.info(f"Все даты успешно распознаны автоматически: {parsed_count}/{total_count}")
                return df
            elif parsed_count > 0:
                logger.warning(f"Частично распознаны даты: {parsed_count}/{total_count}")
        except Exception as e:
            logger.warning(f"Автоматическое распознавание дат не удалось: {e}")
        
        # Если автоматическое распознавание не сработало полностью, пробуем форматы по очереди
        original_dates = df[date_column].copy()
        best_format = None
        best_count = 0
        
        for date_format in date_formats:
            try:
                temp_dates = pd.to_datetime(original_dates, format=date_format, errors='coerce')
                parsed_count = temp_dates.notna().sum()
                
                if parsed_count > best_count:
                    best_count = parsed_count
                    best_format = date_format
                    df[date_column] = temp_dates
                
                if parsed_count == len(df):
                    logger.info(f"Все даты распознаны с форматом {date_format}: {parsed_count}/{len(df)}")
                    break
                    
            except Exception as e:
                continue
        
        # Проверка результата
        final_parsed = df[date_column].notna().sum()
        total = len(df)
        
        if final_parsed == 0:
            logger.error("Не удалось распознать ни одной даты")
            raise ValueError("Не удалось распознать формат дат. Проверьте колонку Datasales")
        elif final_parsed < total:
            logger.warning(f"Распознано только {final_parsed} из {total} дат")
            # Удаляем строки с нераспознанными датами
            df = df.dropna(subset=[date_column])
        else:
            logger.info(f"Успешно распознаны все {final_parsed} дат с форматом {best_format}")
        
        return df
        
    def preprocess_data(self, df):
        """Предобработка данных с улучшенной обработкой дат"""
        df = df.copy()
        
        # Проверка обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        # Улучшенная обработка дат
        df = self.parse_dates(df, 'Datasales')
        
        # Удаление строк с пропущенными критическими значениями
        initial_rows = len(df)
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        
        # Фильтрация аномальных значений
        df = df[df['Price'] > 0]
        df = df[df['Qty'] > 0]
        
        final_rows = len(df)
        if final_rows < initial_rows:
            logger.info(f"Удалено {initial_rows - final_rows} строк с некорректными данными")
        
        if len(df) == 0:
            raise ValueError("После очистки данных не осталось записей")
        
        # Создание временных признаков
        try:
            df['Month'] = df['Datasales'].dt.month
            df['Quarter'] = df['Datasales'].dt.quarter
            df['Weekday'] = df['Datasales'].dt.dayofweek
            df['DayOfMonth'] = df['Datasales'].dt.day
            df['Year'] = df['Datasales'].dt.year
        except Exception as e:
            logger.error(f"Ошибка при создании временных признаков: {e}")
            raise
        
        # Бизнес-метрики
        df['Revenue'] = df['Price'] * df['Qty']
        
        # Безопасное создание категорий цен
        try:
            df['PriceCategory'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        except Exception as e:
            logger.warning(f"Не удалось создать категории цен: {e}")
            df['PriceCategory'] = 'Medium'  # Значение по умолчанию
        
        # Энкодинг с обработкой ошибок
        try:
            df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'].astype(str))
            df['art_encoded'] = self.le_art.fit_transform(df['Art'].astype(str))
        except Exception as e:
            logger.error(f"Ошибка при энкодинге: {e}")
            raise
        
        # Агрегация по магазин-товар с обработкой отсутствующих колонок
        agg_dict = {
            'Qty': ['sum', 'mean', 'count'],
            'Revenue': ['sum', 'mean'],
            'Price': ['mean', 'min', 'max'],
            'Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        }
        
        # Добавляем колонки если они есть
        optional_cols = ['Segment', 'Model', 'Describe']
        for col in optional_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        try:
            agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg(agg_dict).reset_index()
        except Exception as e:
            logger.error(f"Ошибка при агрегации данных: {e}")
            raise
        
        # Упрощение колонок
        new_columns = ['magazin_encoded', 'art_encoded', 'Magazin', 'Art', 
                      'qty_sum', 'qty_mean', 'freq', 'revenue_sum', 'revenue_mean',
                      'price_mean', 'price_min', 'price_max', 'peak_month']
        
        # Добавляем опциональные колонки
        for col in optional_cols:
            if col in df.columns:
                new_columns.append(col)
        
        agg_data.columns = new_columns
        
        # Создание рейтинга с проверкой на валидность
        try:
            rating_components = []
            if 'qty_sum' in agg_data.columns:
                rating_components.append(np.log1p(agg_data['qty_sum']) * 0.4)
            if 'revenue_sum' in agg_data.columns:
                rating_components.append(np.log1p(agg_data['revenue_sum']) * 0.4)
            if 'freq' in agg_data.columns:
                rating_components.append(np.log1p(agg_data['freq']) * 0.2)
            
            if rating_components:
                agg_data['rating'] = sum(rating_components)
                
                # Нормализация рейтинга
                rating_min = agg_data['rating'].min()
                rating_max = agg_data['rating'].max()
                
                if rating_max > rating_min:
                    agg_data['rating'] = (agg_data['rating'] - rating_min) / (rating_max - rating_min) * 4 + 1
                else:
                    agg_data['rating'] = 2.5  # Средний рейтинг если все значения одинаковые
            else:
                agg_data['rating'] = 2.5
                
        except Exception as e:
            logger.warning(f"Ошибка при создании рейтинга: {e}")
            agg_data['rating'] = 2.5
        
        self.processed_data = agg_data
        logger.info(f"Предобработка завершена. Итоговых записей: {len(agg_data)}")
        return agg_data
    
    def create_user_item_matrix(self, df):
        """Создание матрицы пользователь-товар с проверками"""
        try:
            n_users = df['magazin_encoded'].nunique()
            n_items = df['art_encoded'].nunique()
            
            if n_users == 0 or n_items == 0:
                raise ValueError("Нет данных для создания матрицы")
            
            logger.info(f"Создание матрицы {n_users}x{n_items}")
            
            # Создание разреженной матрицы
            user_item_matrix = csr_matrix((df['rating'], 
                                         (df['magazin_encoded'], df['art_encoded'])), 
                                        shape=(n_users, n_items))
            
            self.user_item_matrix = user_item_matrix.toarray()
            
            # Проверка на пустую матрицу
            if np.all(self.user_item_matrix == 0):
                logger.warning("Матрица пользователь-товар пустая")
            
            return self.user_item_matrix
            
        except Exception as e:
            logger.error(f"Ошибка при создании матрицы пользователь-товар: {e}")
            raise
    
    def prepare_content_features(self, df):
        """Подготовка контентных признаков с обработкой ошибок"""
        try:
            # Создание признаков товаров
            agg_dict = {
                'price_mean': 'first',
                'qty_mean': 'first',
                'revenue_mean': 'first'
            }
            
            # Добавляем опциональные колонки
            optional_cols = ['Segment', 'Model']
            for col in optional_cols:
                if col in df.columns:
                    agg_dict[col] = 'first'
            
            item_features = df.groupby('art_encoded').agg(agg_dict).reset_index()
            
            # One-hot encoding для категориальных признаков
            features_list = [item_features[['art_encoded', 'price_mean', 'qty_mean', 'revenue_mean']]]
            
            for col in optional_cols:
                if col in item_features.columns:
                    try:
                        dummies = pd.get_dummies(item_features[col], prefix=col.lower())
                        features_list.append(dummies)
                    except Exception as e:
                        logger.warning(f"Не удалось создать dummy-переменные для {col}: {e}")
            
            # Объединение признаков
            features = pd.concat(features_list, axis=1)
            
            self.feature_columns = [col for col in features.columns if col != 'art_encoded']
            
            # Нормализация числовых признаков
            numeric_cols = ['price_mean', 'qty_mean', 'revenue_mean']
            existing_numeric_cols = [col for col in numeric_cols if col in features.columns]
            
            if existing_numeric_cols:
                features[existing_numeric_cols] = self.scaler.fit_transform(features[existing_numeric_cols])
            
            logger.info(f"Подготовлены признаки для {len(features)} товаров")
            return features
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке контентных признаков: {e}")
            return pd.DataFrame()
    
    def build_ensemble_model(self, df, test_size=0.2):
        """Построение ансамбля моделей с улучшенной обработкой ошибок"""
        try:
            # Предобработка
            df = self.preprocess_data(df)
            user_item_matrix = self.create_user_item_matrix(df)
            content_features = self.prepare_content_features(df)
            
            if len(df) < 10:
                raise ValueError("Недостаточно данных для обучения модели (минимум 10 записей)")
            
            # Разделение данных
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
            
            models_trained = []
            
            # 1. SVD (Matrix Factorization)
            try:
                n_components = min(50, min(user_item_matrix.shape) - 1)
                if n_components > 0:
                    self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                    svd_matrix = self.svd_model.fit_transform(user_item_matrix)
                    models_trained.append('SVD')
                    logger.info(f"SVD модель обучена с {n_components} компонентами")
            except Exception as e:
                logger.warning(f"Не удалось обучить SVD модель: {e}")
            
            # 2. NMF (Non-negative Matrix Factorization)
            try:
                n_components = min(30, min(user_item_matrix.shape) - 1)
                if n_components > 0:
                    self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=500)
                    nmf_matrix = self.nmf_model.fit_transform(user_item_matrix)
                    models_trained.append('NMF')
                    logger.info(f"NMF модель обучена с {n_components} компонентами")
            except Exception as e:
                logger.warning(f"Не удалось обучить NMF модель: {e}")
            
            # 3. Item-based Collaborative Filtering
            try:
                self.item_similarity = cosine_similarity(user_item_matrix.T)
                models_trained.append('ItemCF')
                logger.info("Item-based CF модель обучена")
            except Exception as e:
                logger.warning(f"Не удалось обучить Item-based CF: {e}")
            
            # 4. Content-based Random Forest
            try:
                if len(content_features) > 0 and len(self.feature_columns) > 0:
                    rf_data = df.merge(content_features, on='art_encoded', how='left')
                    X_rf = rf_data[self.feature_columns].fillna(0)
                    y_rf = rf_data['rating']
                    
                    if len(X_rf) > 0 and len(y_rf) > 0:
                        self.rf_model = RandomForestRegressor(
                            n_estimators=100, 
                            random_state=42, 
                            max_depth=10,
                            min_samples_split=5
                        )
                        self.rf_model.fit(X_rf, y_rf)
                        models_trained.append('RandomForest')
                        logger.info("Random Forest модель обучена")
            except Exception as e:
                logger.warning(f"Не удалось обучить Random Forest: {e}")
            
            if not models_trained:
                raise ValueError("Не удалось обучить ни одной модели")
            
            # Вычисление метрик
            try:
                train_predictions = self.predict_ratings_for_evaluation(train_data)
                test_predictions = self.predict_ratings_for_evaluation(test_data)
                
                train_rmse = np.sqrt(np.mean((train_data['rating'] - train_predictions) ** 2))
                test_rmse = np.sqrt(np.mean((test_data['rating'] - test_predictions) ** 2))
            except Exception as e:
                logger.warning(f"Не удалось вычислить метрики: {e}")
                train_rmse = test_rmse = 0.0
            
            result = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'n_users': len(df['magazin_encoded'].unique()),
                'n_items': len(df['art_encoded'].unique()),
                'sparsity': 1 - np.count_nonzero(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]),
                'models_trained': models_trained
            }
            
            logger.info(f"Ансамбль обучен. Модели: {models_trained}")
            return result
            
        except Exception as e:
            logger.error(f"Критическая ошибка при обучении модели: {e}")
            raise
    
    def predict_ratings_for_evaluation(self, test_data):
        """Предсказание рейтингов для оценки модели"""
        predictions = []
        
        for _, row in test_data.iterrows():
            try:
                user_id = row['magazin_encoded']
                item_id = row['art_encoded']
                pred = self.predict_single_rating(user_id, item_id)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Ошибка предсказания для пользователя {user_id}, товара {item_id}: {e}")
                predictions.append(2.5)  # Средний рейтинг по умолчанию
        
        return np.array(predictions)
    
    def predict_single_rating(self, user_id, item_id):
        """Предсказание одного рейтинга с обработкой ошибок"""
        predictions = []
        
        # Проверка валидности индексов
        if (self.user_item_matrix is None or 
            user_id >= self.user_item_matrix.shape[0] or 
            item_id >= self.user_item_matrix.shape[1] or
            user_id < 0 or item_id < 0):
            return 2.5
        
        # SVD предсказание
        if self.svd_model is not None:
            try:
                user_factors = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('svd', svd_pred))
            except Exception as e:
                logger.debug(f"SVD предсказание не удалось: {e}")
        
        # NMF предсказание
        if self.nmf_model is not None:
            try:
                user_factors = self.nmf_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.nmf_model.components_[:, item_id]
                nmf_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('nmf', nmf_pred))
            except Exception as e:
                logger.debug(f"NMF предсказание не удалось: {e}")
        
        # Item similarity предсказание
        if self.item_similarity is not None:
            try:
                user_ratings = self.user_item_matrix[user_id]
                similar_items = self.item_similarity[item_id]
                
                # Взвешенное среднее по похожим товарам
                numerator = np.sum(similar_items * user_ratings)
                denominator = np.sum(np.abs(similar_items))
                
                if denominator > 0:
                    similarity_pred = numerator / denominator
                    predictions.append(('similarity', similarity_pred))
            except Exception as e:
                logger.debug(f"Similarity предсказание не удалось: {e}")
        
        # Ансамблевое предсказание
        if predictions:
            try:
                weighted_sum = sum(pred * self.weights.get(method, 0.25) for method, pred in predictions)
                total_weight = sum(self.weights.get(method, 0.25) for method, _ in predictions)
                return weighted_sum / total_weight if total_weight > 0 else 2.5
            except Exception as e:
                logger.debug(f"Ансамблевое предсказание не удалось: {e}")
        
        return 2.5  # Средний рейтинг по умолчанию
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Получение рекомендаций для магазина с обработкой ошибок"""
        if self.user_item_matrix is None:
            logger.warning("Модель не обучена")
            return None
        
        try:
            user_id = self.le_magazin.transform([magazin_name])[0]
        except ValueError:
            logger.warning(f"Магазин {magazin_name} не найден")
            return None
        except Exception as e:
            logger.error(f"Ошибка при поиске магазина: {e}")
            return None
        
        if user_id >= self.user_item_matrix.shape[0]:
            logger.warning(f"Неверный ID пользователя: {user_id}")
            return None
        
        try:
            # Получение всех товаров
            n_items = self.user_item_matrix.shape[1]
            user_ratings = self.user_item_matrix[user_id]
            
            # Предсказание рейтингов для всех товаров
            predictions = []
            for item_id in range(n_items):
                if user_ratings[item_id] == 0:  # Только для неоцененных товаров
                    pred_rating = self.predict_single_rating(user_id, item_id)
                    predictions.append((item_id, pred_rating))
            
            # Сортировка и выбор топ-K
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_items = predictions[:top_k]
            
            # Формирование рекомендаций
            recommendations = []
            for rank, (item_id, score) in enumerate(top_items, 1):
                try:
                    item_name = self.le_art.inverse_transform([item_id])[0]
                    
                    # Поиск информации о товаре
                    item_info = self.processed_data[
                        self.processed_data['art_encoded'] == item_id
                    ]
                    
                    if len(item_info) > 0:
                        info = item_info.iloc[0]
                        rec = {
                            'rank': rank,
                            'item': item_name,
                            'score': score,
                            'segment': info.get('Segment', 'Unknown'),
                            'model': info.get('Model', 'Unknown'),
                            'avg_price': info.get('price_mean', 0),
                            'expected_qty': info.get('qty_mean', 0)
                        }
                    else:
                        rec = {
                            'rank': rank,
                            'item': item_name,
                            'score': score,
                            'segment': 'Unknown',
                            'model': 'Unknown',
                            'avg_price': 0,
                            'expected_qty': 0
                        }
                    
                    recommendations.append(rec)
                except Exception as e:
                    logger.warning(f"Ошибка при формировании рекомендации для товара {item_id}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {e}")
            return None
    
    def get_all_recommendations(self, top_k=10):
        """Получение рекомендаций для всех магазинов"""
        if self.user_item_matrix is None:
            return None
        
        all_recommendations = {}
        for magazin_name in self.le_magazin.classes_:
            try:
                recommendations = self.get_recommendations(magazin_name, top_k)
                if recommendations:
                    all_recommendations[magazin_name] = recommendations
            except Exception as e:
                logger.warning(f"Ошибка при получении рекомендаций для {magazin_name}: {e}")
                continue
        
        return all_recommendations

def create_dashboard():
    st.set_page_config(page_title="Рекомендательная система", layout="wide")
    
    st.title("🛍️ Рекомендательная система для магазинов")
    st.markdown("*Ансамбль алгоритмов: SVD + NMF + Коллаборативная фильтрация + Content-based*")
    st.markdown("---")
    
    # Инициализация системы
    if 'recommender' not in st.session_state:
        st.session_state.recommender = EnsembleRecommenderSystem()
    
    # Загрузка файла
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите Excel файл", 
        type=['xlsx', 'xls'],
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Price, Qty (обязательные) + Describe, Model, Segment (опциональные)"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Проверка критически важных колонок
            required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Отсутствуют обязательные колонки: {missing_cols}")
                st.info("📋 Обязательные колонки: Magazin, Datasales, Art, Price, Qty")
                return
            
            # Отображение информации о данных
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Записей", len(df))
            with col2:
                st.metric("Магазинов", df['Magazin'].nunique())
            with col3:
                st.metric("Товаров", df['Art'].nunique())
            with col4:
                if 'Segment' in df.columns:
                    st.metric("Сегментов", df['Segment'].nunique())
                else:
                    st.metric("Сегментов", "Н/Д")
            
            # Показать примеры дат для проверки
            st.expander("🔍 Предварительный просмотр дат").write(
                df['Datasales'].head(10).to_list()
            )
            
            # Построение модели
            if st.sidebar.button("🚀 Построить модель", type="primary"):
                try:
                    with st.spinner("Обучение ансамбля моделей..."):
                        metrics = st.session_state.recommender.build_ensemble_model(df)
                    
                    st.success("✅ Ансамбль моделей успешно обучен!")
                    
                    # Показать какие модели обучились
                    if 'models_trained' in metrics:
                        models_info = ", ".join(metrics['models_trained'])
                        st.info(f"🤖 Обученные модели: {models_info}")
                    
                    # Метрики модели
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE (train)", f"{metrics['train_rmse']:.3f}")
                    with col2:
                        st.metric("RMSE (test)", f"{metrics['test_rmse']:.3f}")
                    with col3:
                        st.metric("Разреженность", f"{metrics['sparsity']:.1%}")
                    with col4:
                        overfitting = metrics['test_rmse'] - metrics['train_rmse']
                        st.metric("Переобучение", f"{overfitting:.3f}")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обучении модели: {str(e)}")
                    st.info("💡 Проверьте корректность данных и формат файла")
            
            # Рекомендации
            if st.session_state.recommender.user_item_matrix is not None:
                st.markdown("---")
                st.header("📊 Рекомендации")
                
                tab1, tab2, tab3 = st.tabs(["Для одного магазина", "Для всех магазинов", "Аналитика"])
                
                with tab1:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        selected_shop = st.selectbox(
                            "Выберите магазин:",
                            options=st.session_state.recommender.le_magazin.classes_
                        )
                    with col2:
                        top_k = st.slider("Количество рекомендаций:", 5, 20, 10)
                    
                    if st.button("Получить рекомендации"):
                        try:
                            recommendations = st.session_state.recommender.get_recommendations(selected_shop, top_k)
                            
                            if recommendations:
                                rec_df = pd.DataFrame(recommendations)
                                
                                # Форматирование для отображения
                                display_df = rec_df.copy()
                                display_df['score'] = display_df['score'].round(3)
                                display_df['avg_price'] = display_df['avg_price'].round(2)
                                display_df['expected_qty'] = display_df['expected_qty'].round(1)
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # График рекомендаций
                                fig = px.bar(
                                    rec_df.head(10), x='item', y='score',
                                    title=f"Топ-10 рекомендаций для {selected_shop}",
                                    labels={'item': 'Товар', 'score': 'Прогнозный рейтинг'},
                                    color='score',
                                    color_continuous_scale='viridis'
                                )
                                fig.update_xaxes(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # График по сегментам (если есть данные)
                                if 'segment' in rec_df.columns and rec_df['segment'].nunique() > 1:
                                    segment_counts = rec_df['segment'].value_counts()
                                    fig2 = px.pie(
                                        values=segment_counts.values,
                                        names=segment_counts.index,
                                        title="Распределение рекомендаций по сегментам"
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.warning("⚠️ Не удалось сгенерировать рекомендации для выбранного магазина")
                                
                        except Exception as e:
                            st.error(f"❌ Ошибка при получении рекомендаций: {str(e)}")
                
                with tab2:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        batch_top_k = st.slider("Рекомендаций на магазин:", 5, 15, 10)
                    with col2:
                        show_top_n = st.slider("Показать топ для отчета:", 3, 10, 5)
                    
                    if st.button("Сгенерировать рекомендации для всех магазинов"):
                        try:
                            with st.spinner("Генерация рекомендаций..."):
                                all_recs = st.session_state.recommender.get_all_recommendations(batch_top_k)
                            
                            if all_recs:
                                # Создание сводной таблицы
                                summary_data = []
                                for shop, recs in all_recs.items():
                                    for rec in recs[:show_top_n]:
                                        summary_data.append({
                                            'Магазин': shop,
                                            'Ранг': rec['rank'],
                                            'Товар': rec['item'],
                                            'Прогноз': f"{rec['score']:.3f}",
                                            'Сегмент': rec['segment'],
                                            'Модель': rec['model'],
                                            'Средняя цена': f"{rec['avg_price']:.2f}",
                                            'Ожидаемое кол-во': f"{rec['expected_qty']:.1f}"
                                        })
                                
                                if summary_data:
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df, use_container_width=True)
                                    
                                    # Статистика по рекомендациям
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Всего рекомендаций", len(summary_data))
                                    with col2:
                                        avg_score = np.mean([float(x) for x in summary_df['Прогноз']])
                                        st.metric("Средний прогноз", f"{avg_score:.3f}")
                                    with col3:
                                        unique_items = summary_df['Товар'].nunique()
                                        st.metric("Уникальных товаров", unique_items)
                                    
                                    # Скачивание результатов
                                    @st.cache_data
                                    def convert_df(df):
                                        return df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                                    
                                    csv = convert_df(summary_df)
                                    st.download_button(
                                        label="📥 Скачать рекомендации (CSV)",
                                        data=csv,
                                        file_name='ensemble_recommendations.csv',
                                        mime='text/csv'
                                    )
                                else:
                                    st.warning("⚠️ Не удалось создать сводную таблицу рекомендаций")
                            else:
                                st.warning("⚠️ Не удалось сгенерировать рекомендации")
                                
                        except Exception as e:
                            st.error(f"❌ Ошибка при генерации рекомендаций: {str(e)}")
                
                with tab3:
                    if st.session_state.recommender.processed_data is not None:
                        try:
                            data = st.session_state.recommender.processed_data
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Распределение рейтингов
                                fig1 = px.histogram(
                                    data, x='rating', bins=20,
                                    title="Распределение рейтингов товаров",
                                    labels={'rating': 'Рейтинг', 'count': 'Количество'}
                                )
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            with col2:
                                # Сегменты по рейтингу (если есть)
                                if 'Segment' in data.columns:
                                    segment_rating = data.groupby('Segment')['rating'].mean().sort_values(ascending=False)
                                    fig2 = px.bar(
                                        x=segment_rating.index, y=segment_rating.values,
                                        title="Средний рейтинг по сегментам",
                                        labels={'x': 'Сегмент', 'y': 'Средний рейтинг'}
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                                else:
                                    # Альтернативный график - топ товаров
                                    top_items = data.nlargest(10, 'rating')
                                    fig2 = px.bar(
                                        top_items, x='Art', y='rating',
                                        title="Топ-10 товаров по рейтингу"
                                    )
                                    fig2.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig2, use_container_width=True)
                            
                            # Корреляционная матрица
                            numeric_cols = ['qty_sum', 'revenue_sum', 'price_mean', 'freq', 'rating']
                            available_cols = [col for col in numeric_cols if col in data.columns]
                            
                            if len(available_cols) > 1:
                                corr_matrix = data[available_cols].corr()
                                
                                fig3 = px.imshow(
                                    corr_matrix, 
                                    title="Корреляции между метриками",
                                    aspect="auto",
                                    color_continuous_scale='RdBu',
                                    text_auto=True
                                )
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Дополнительная статистика
                            st.subheader("📈 Дополнительная статистика")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Средний рейтинг", f"{data['rating'].mean():.2f}")
                            with col2:
                                st.metric("Медианный рейтинг", f"{data['rating'].median():.2f}")
                            with col3:
                                st.metric("Мин рейтинг", f"{data['rating'].min():.2f}")
                            with col4:
                                st.metric("Макс рейтинг", f"{data['rating'].max():.2f}")
                                
                        except Exception as e:
                            st.error(f"❌ Ошибка при создании аналитики: {str(e)}")
                    else:
                        st.info("📊 Аналитика будет доступна после обучения модели")
        
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {str(e)}")
            st.info("💡 Убедитесь, что файл содержит все необходимые колонки и корректные данные")
    
    else:
        st.info("👆 Загрузите Excel файл для начала работы")
        
        # Пример структуры данных
        st.markdown("### 📋 Пример структуры данных:")
        example_data = {
            'Magazin': ['Shop_A', 'Shop_B', 'Shop_A', 'Shop_C'],
            'Datasales': ['2024-01-15', '16.01.2024', '17/01/2024', '2024-01-18'],
            'Art': ['Item_001', 'Item_002', 'Item_003', 'Item_001'],
            'Describe': ['Описание 1', 'Описание 2', 'Описание 3', 'Описание 1'],
            'Model': ['Model_X', 'Model_Y', 'Model_Z', 'Model_X'],
            'Segment': ['Electronics', 'Clothing', 'Electronics', 'Electronics'],
            'Price': [100, 50, 150, 105],
            'Qty': [2, 1, 3, 1],
            'Sum': [200, 50, 450, 105]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("### 🔧 Технические особенности:")
        st.markdown("""
        - **Улучшенная обработка дат**: Автоматическое распознавание 14+ форматов дат
        - **SVD**: Матричная факторизация для скрытых паттернов
        - **NMF**: Неотрицательная факторизация для интерпретируемости  
        - **Item-based CF**: Рекомендации на основе похожести товаров
        - **Content-based**: Учет характеристик товаров через Random Forest
        - **Ансамблирование**: Взвешенное объединение всех подходов
        - **Обработка ошибок**: Робустность к некорректным данным
        """)
        
        st.markdown("### 📅 Поддерживаемые форматы дат:")
        date_formats_info = """
        - `YYYY-MM-DD` (2024-01-15)
        - `DD.MM.YYYY` (15.01.2024)
        - `DD/MM/YYYY` (15/01/2024)
        - `MM/DD/YYYY` (01/15/2024)
        - `DD-MM-YYYY` (15-01-2024)
        - И другие популярные форматы
        """
        st.markdown(date_formats_info)

if __name__ == "__main__":
    create_dashboard()
