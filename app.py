import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class OptimizedRecommenderSystem:
    def __init__(self):
        self.svd_model = None
        self.rf_model = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.scaler = StandardScaler()
        self.processed_data = None
        self.content_features = None
        self.use_rf = False
        self.weights = {'svd': 0.5, 'similarity': 0.3, 'rf': 0.2}
        
        # Новые параметры для фильтрации
        self.min_sales_filter = 2  # Минимальное количество продаж по умолчанию
        self.item_popularity = {}  # Популярность товаров
        self.eligible_items = set()  # Товары, прошедшие фильтр
        
    def process_datasales(self, df):
        """Обработка колонки Datasales с различными вариантами"""
        if 'Datasales' not in df.columns:
            return df
        
        # Попытка автоматического определения формата даты
        datasales_col = df['Datasales'].copy()
        
        # Удаляем пустые значения для анализа
        non_null_dates = datasales_col.dropna()
        if len(non_null_dates) == 0:
            return df
        
        # Пробуем разные варианты парсинга
        date_formats = [
            '%Y-%m-%d',
            '%d.%m.%Y', 
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%Y.%m.%d'
        ]
        
        parsed_dates = None
        successful_format = None
        
        # Принудительное приведение к datetime64
        try:
            parsed_dates = pd.to_datetime(datasales_col, errors='coerce')
            if parsed_dates.notna().sum() > len(non_null_dates) * 0.8:  # Если 80%+ успешно
                df['Datasales'] = parsed_dates.astype('datetime64[ns]')
                successful_format = "auto"
        except:
            pass
        
        # Если автоматический парсинг не сработал, пробуем форматы
        if successful_format is None:
            for fmt in date_formats:
                try:
                    test_dates = pd.to_datetime(non_null_dates.iloc[:min(100, len(non_null_dates))], 
                                              format=fmt, errors='coerce')
                    success_rate = test_dates.notna().sum() / len(test_dates)
                    
                    if success_rate > 0.8:  # Если 80%+ успешно распознано
                        parsed_dates = pd.to_datetime(datasales_col, format=fmt, errors='coerce')
                        df['Datasales'] = parsed_dates.astype('datetime64[ns]')
                        successful_format = fmt
                        break
                except:
                    continue
        
        # Если ничего не сработало, пробуем числовые значения как timestamp
        if successful_format is None:
            try:
                # Проверяем, не являются ли значения timestamp'ами
                numeric_dates = pd.to_numeric(datasales_col, errors='coerce')
                if numeric_dates.notna().sum() > 0:
                    # Пробуем интерпретировать как секунды или миллисекунды
                    test_val = numeric_dates.dropna().iloc[0]
                    if test_val > 1e9:  # Похоже на timestamp в секундах или миллисекундах
                        if test_val > 1e12:  # Миллисекунды
                            parsed_dates = pd.to_datetime(numeric_dates, unit='ms', errors='coerce')
                        else:  # Секунды
                            parsed_dates = pd.to_datetime(numeric_dates, unit='s', errors='coerce')
                        
                        df['Datasales'] = parsed_dates.astype('datetime64[ns]')
                        successful_format = "timestamp"
            except:
                pass
        
        # Добавляем временные признаки, если дата успешно обработана
        if successful_format and 'Datasales' in df.columns and df['Datasales'].dtype == 'datetime64[ns]':
            df['Month'] = df['Datasales'].dt.month
            df['Quarter'] = df['Datasales'].dt.quarter
            df['Weekday'] = df['Datasales'].dt.dayofweek
            df['DayOfMonth'] = df['Datasales'].dt.day
            df['Year'] = df['Datasales'].dt.year
            
            # Информация об успешной обработке
            st.info(f"✅ Колонка Datasales обработана (формат: {successful_format})")
        else:
            st.warning("⚠️ Не удалось обработать колонку Datasales. Продолжаем без временных признаков.")
        
        return df
    
    def calculate_item_popularity(self, df):
        """Расчет популярности товаров - количество уникальных транзакций и магазинов"""
        # Подсчитываем количество уникальных транзакций (строк) для каждого товара
        item_transaction_count = df.groupby('Art').size().to_dict()
        
        # Подсчитываем количество уникальных магазинов для каждого товара
        item_store_count = df.groupby('Art')['Magazin'].nunique().to_dict()
        
        # Суммарное количество проданных единиц
        item_total_qty = df.groupby('Art')['Qty'].sum().to_dict()
        
        # Комбинированная популярность: транзакции + магазины + общее количество
        self.item_popularity = {}
        for art in df['Art'].unique():
            # Взвешенная популярность
            popularity_score = (
                item_transaction_count.get(art, 0) * 0.4 +  # 40% вес транзакций
                item_store_count.get(art, 0) * 0.4 +        # 40% вес магазинов
                min(item_total_qty.get(art, 0) / 10, 10) * 0.2  # 20% вес общего количества (ограничено)
            )
            self.item_popularity[art] = {
                'transactions': item_transaction_count.get(art, 0),
                'stores': item_store_count.get(art, 0),
                'total_qty': item_total_qty.get(art, 0),
                'popularity_score': popularity_score
            }
    
    def filter_eligible_items(self, min_transactions=None):
        """Фильтрация товаров по минимальному количеству транзакций"""
        if min_transactions is None:
            min_transactions = self.min_sales_filter
        
        self.eligible_items = set()
        filtered_count = 0
        
        for art, stats in self.item_popularity.items():
            if stats['transactions'] >= min_transactions:
                self.eligible_items.add(art)
            else:
                filtered_count += 1
        
        return len(self.eligible_items), filtered_count
    
    def process_data(self, df):
        """Упрощенная предобработка данных с фильтрацией"""
        df = df.copy()
        
        # Обработка колонки Datasales если она есть
        df = self.process_datasales(df)
        
        # Базовая очистка
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        df = df[df['Price'] > 0]
        df = df[df['Qty'] > 0]
        
        # Приведение к строковому типу для энкодинга
        df['Magazin'] = df['Magazin'].astype(str)
        df['Art'] = df['Art'].astype(str)
        
        # Расчет популярности товаров ПЕРЕД агрегацией
        self.calculate_item_popularity(df)
        
        # Создание рейтинга на основе количества и выручки
        df['Revenue'] = df['Price'] * df['Qty']
        
        # Агрегация по магазин-товар
        agg_data = df.groupby(['Magazin', 'Art']).agg({
            'Qty': 'sum',
            'Revenue': 'sum',
            'Price': 'mean',
            'Segment': 'first',
            'Model': 'first'
        }).reset_index()
        
        # Энкодинг ПОСЛЕ агрегации
        agg_data['magazin_encoded'] = self.le_magazin.fit_transform(agg_data['Magazin'])
        agg_data['art_encoded'] = self.le_art.fit_transform(agg_data['Art'])
        
        # Убеждаемся, что все поля имеют правильный тип
        agg_data['Segment'] = agg_data['Segment'].astype(str)
        agg_data['Model'] = agg_data['Model'].astype(str)
        
        # Создание простого рейтинга
        agg_data['rating'] = np.log1p(agg_data['Qty']) + np.log1p(agg_data['Revenue']) * 0.1
        
        # Нормализация рейтинга от 1 до 5
        min_rating = agg_data['rating'].min()
        max_rating = agg_data['rating'].max()
        if max_rating > min_rating:
            agg_data['rating'] = (agg_data['rating'] - min_rating) / (max_rating - min_rating) * 4 + 1
        else:
            agg_data['rating'] = 2.5
        
        self.processed_data = agg_data
        return agg_data
    
    def create_user_item_matrix(self, df):
        """Создание разреженной матрицы пользователь-товар"""
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        # Создание разреженной матрицы
        sparse_matrix = csr_matrix((df['rating'], 
                                  (df['magazin_encoded'], df['art_encoded'])), 
                                 shape=(n_users, n_items))
        
        self.user_item_matrix = sparse_matrix
        return sparse_matrix
    
    def build_model(self, df, test_size=0.2, min_sales_filter=2):
        """Построение упрощенной модели с фильтрацией"""
        self.min_sales_filter = min_sales_filter
        
        # Предобработка
        df = self.process_data(df)
        
        # Фильтрация товаров по популярности
        eligible_count, filtered_count = self.filter_eligible_items(min_sales_filter)
        
        user_item_matrix = self.create_user_item_matrix(df)
        
        # Разделение данных для оценки
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Преобразование в плотную матрицу для SVD
        dense_matrix = user_item_matrix.toarray()
        
        # SVD (Matrix Factorization)
        n_components = min(30, min(dense_matrix.shape) - 1)
        if n_components <= 0:
            n_components = 1
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(dense_matrix)
        
        # Item-based Collaborative Filtering (только для топ товаров)
        item_counts = np.array(user_item_matrix.sum(axis=0))[0]
        
        if len(item_counts) > 0 and np.max(item_counts) > 0:
            top_items_mask = item_counts >= np.percentile(item_counts[item_counts > 0], 50)
            
            if np.sum(top_items_mask) > 0:
                filtered_matrix = dense_matrix[:, top_items_mask]
                if filtered_matrix.shape[1] > 1:
                    self.item_similarity = cosine_similarity(filtered_matrix.T)
                    self.top_items_indices = np.where(top_items_mask)[0]
                else:
                    self.item_similarity = None
                    self.top_items_indices = None
            else:
                if dense_matrix.shape[1] > 1:
                    self.item_similarity = cosine_similarity(dense_matrix.T)
                    self.top_items_indices = np.arange(dense_matrix.shape[1])
                else:
                    self.item_similarity = None
                    self.top_items_indices = None
        else:
            self.item_similarity = None
            self.top_items_indices = None
        
        # Быстрая оценка качества
        sample_size = min(1000, len(test_data))
        if sample_size > 0:
            test_sample = test_data.sample(n=sample_size, random_state=42)
            
            predictions = []
            actuals = []
            
            for _, row in test_sample.iterrows():
                pred = self.predict_single_rating(row['magazin_encoded'], row['art_encoded'])
                predictions.append(pred)
                actuals.append(row['rating'])
            
            if len(predictions) > 0:
                rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))
            else:
                rmse = 0.0
        else:
            rmse = 0.0
        
        return {
            'rmse': rmse,
            'n_users': len(df['magazin_encoded'].unique()),
            'n_items': len(df['art_encoded'].unique()),
            'sparsity': 1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1]),
            'eligible_items': eligible_count,
            'filtered_items': filtered_count,
            'min_sales_filter': min_sales_filter
        }
    
    def is_item_eligible(self, item_name):
        """Проверка, проходит ли товар фильтр популярности"""
        return item_name in self.eligible_items
    
    def predict_single_rating(self, user_id, item_id):
        """Быстрое предсказание рейтинга"""
        predictions = []
        dense_matrix = self.user_item_matrix.toarray()
        
        # SVD предсказание
        if self.svd_model and user_id < dense_matrix.shape[0] and item_id < dense_matrix.shape[1]:
            try:
                user_factors = self.svd_model.transform(dense_matrix[user_id:user_id+1])
                item_factors = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_factors[0], item_factors)
                predictions.append(('svd', svd_pred))
            except:
                pass
        
        # Item similarity предсказание
        if (self.item_similarity is not None and 
            hasattr(self, 'top_items_indices') and 
            self.top_items_indices is not None and
            user_id < dense_matrix.shape[0] and 
            item_id in self.top_items_indices):
            
            try:
                item_idx_in_filtered = np.where(self.top_items_indices == item_id)[0]
                if len(item_idx_in_filtered) > 0:
                    item_idx = item_idx_in_filtered[0]
                    user_ratings = dense_matrix[user_id, self.top_items_indices]
                    similar_items = self.item_similarity[item_idx]
                    
                    # Простое взвешенное среднее
                    mask = user_ratings > 0
                    if np.sum(mask) > 0:
                        numerator = np.sum(similar_items[mask] * user_ratings[mask])
                        denominator = np.sum(np.abs(similar_items[mask]))
                        
                        if denominator > 0:
                            similarity_pred = numerator / denominator
                            predictions.append(('similarity', similarity_pred))
            except:
                pass
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights[method] for method, pred in predictions)
            total_weight = sum(self.weights[method] for method, _ in predictions)
            return weighted_sum / total_weight if total_weight > 0 else 2.5
        
        return 2.5
    
    def get_recommendations(self, magazin_name, top_k=10, apply_popularity_filter=True):
        """Получение рекомендаций с фильтрацией по популярности"""
        if self.user_item_matrix is None:
            return None
        
        try:
            user_id = self.le_magazin.transform([magazin_name])[0]
        except:
            return None
        
        dense_matrix = self.user_item_matrix.toarray()
        if user_id >= dense_matrix.shape[0]:
            return None
        
        # Получение товаров, которые пользователь еще не покупал
        user_ratings = dense_matrix[user_id]
        unrated_items = np.where(user_ratings == 0)[0]
        
        if len(unrated_items) == 0:
            return []
        
        # Предсказание только для непокупанных товаров
        predictions = []
        for item_id in unrated_items:
            try:
                item_name = self.le_art.inverse_transform([item_id])[0]
                
                # Применяем фильтр популярности если включен
                if apply_popularity_filter and not self.is_item_eligible(item_name):
                    continue
                
                pred_rating = self.predict_single_rating(user_id, item_id)
                predictions.append((item_id, pred_rating, item_name))
            except:
                continue
        
        # Сортировка и выбор топ-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = predictions[:top_k]
        
        # Формирование результата
        recommendations = []
        for rank, (item_id, score, item_name) in enumerate(top_items, 1):
            try:
                # Поиск информации о товаре
                item_info = self.processed_data[self.processed_data['art_encoded'] == item_id]
                
                if len(item_info) > 0:
                    info = item_info.iloc[0]
                    # Добавляем статистику популярности
                    popularity_stats = self.item_popularity.get(item_name, {})
                    
                    recommendations.append({
                        'rank': rank,
                        'item': item_name,
                        'score': round(score, 3),
                        'segment': info['Segment'],
                        'model': info['Model'],
                        'avg_price': round(info['Price'], 2),
                        'total_qty': int(info['Qty']),
                        'transactions': popularity_stats.get('transactions', 0),
                        'stores': popularity_stats.get('stores', 0),
                        'popularity_score': round(popularity_stats.get('popularity_score', 0), 2)
                    })
                else:
                    popularity_stats = self.item_popularity.get(item_name, {})
                    recommendations.append({
                        'rank': rank,
                        'item': item_name,
                        'score': round(score, 3),
                        'segment': 'Unknown',
                        'model': 'Unknown',
                        'avg_price': 0,
                        'total_qty': 0,
                        'transactions': popularity_stats.get('transactions', 0),
                        'stores': popularity_stats.get('stores', 0),
                        'popularity_score': round(popularity_stats.get('popularity_score', 0), 2)
                    })
            except:
                continue
        
        return recommendations
    
    def get_all_recommendations(self, top_k=10, apply_popularity_filter=True):
        """Получение рекомендаций для всех магазинов"""
        if self.user_item_matrix is None:
            return None
        
        all_recommendations = {}
        for magazin_name in self.le_magazin.classes_:
            recommendations = self.get_recommendations(magazin_name, top_k, apply_popularity_filter)
            if recommendations:
                all_recommendations[magazin_name] = recommendations
        
        return all_recommendations
    
    def get_popularity_stats(self):
        """Получение статистики популярности товаров"""
        if not self.item_popularity:
            return None
        
        stats_df = pd.DataFrame.from_dict(self.item_popularity, orient='index')
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'item'}, inplace=True)
        
        return stats_df.sort_values('popularity_score', ascending=False)

def create_dashboard():
    st.set_page_config(page_title="Рекомендательная система", layout="wide")
    
    st.title("🛍️ Улучшенная рекомендательная система")
    st.markdown("*SVD + Item-based коллаборативная фильтрация с фильтром популярности*")
    st.markdown("---")
    
    # Инициализация системы
    if 'recommender' not in st.session_state:
        st.session_state.recommender = OptimizedRecommenderSystem()
    
    # Загрузка файла
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите Excel файл", 
        type=['xlsx', 'xls'],
        help="Файл должен содержать колонки: Magazin, Art, Segment, Model, Price, Qty"
    )
    
    # Настройки фильтрации
    st.sidebar.header("⚙️ Настройки фильтрации")
    min_sales_filter = st.sidebar.number_input(
        "Минимальное количество транзакций по сети:",
        min_value=1,
        max_value=50,
        value=2,
        help="Товары с меньшим количеством транзакций будут исключены из рекомендаций"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Предварительная очистка типов данных
            st.info("🔄 Предварительная обработка данных...")
            
            # Проверка и приведение к правильным типам
            if 'Price' in df.columns:
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            if 'Qty' in df.columns:
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
            
            # Удаление строк с некорректными данными после приведения типов
            df = df.dropna(subset=['Price', 'Qty'])
            
            # Проверка колонок
            required_cols = ['Magazin', 'Art', 'Segment', 'Model', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
                return
            
            # Проверка на пустые данные после очистки
            if len(df) == 0:
                st.error("После очистки данных не осталось валидных записей. Проверьте качество данных.")
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
                st.metric("Сегментов", df['Segment'].nunique())
            
            # Построение модели
            if st.sidebar.button("🚀 Построить модель", type="primary"):
                with st.spinner("Обучение модели..."):
                    metrics = st.session_state.recommender.build_model(df, min_sales_filter=min_sales_filter)
                
                st.success("Модель обучена!")
                
                # Метрики модели
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                with col2:
                    st.metric("Разреженность", f"{metrics['sparsity']:.1%}")
                with col3:
                    st.metric("Прошли фильтр", f"{metrics['eligible_items']}")
                with col4:
                    st.metric("Отфильтровано", f"{metrics['filtered_items']}")
                
                # Дополнительная информация о фильтрации
                st.info(f"🔍 Применен фильтр: минимум {metrics['min_sales_filter']} транзакций по сети")
            
            # Статистика популярности товаров
            if st.session_state.recommender.item_popularity:
                st.markdown("---")
                st.header("📈 Статистика популярности товаров")
                
                if st.button("Показать статистику популярности"):
                    stats_df = st.session_state.recommender.get_popularity_stats()
                    if stats_df is not None:
                        # Добавим цветовое кодирование
                        st.dataframe(
                            stats_df.head(50),  # Показываем топ 50
                            use_container_width=True
                        )
                        
                        # Статистика фильтрации
                        total_items = len(stats_df)
                        eligible_items = len(stats_df[stats_df['transactions'] >= min_sales_filter])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Всего товаров", total_items)
                        with col2:
                            st.metric("Прошли фильтр", eligible_items)
                        with col3:
                            filter_ratio = (eligible_items / total_items * 100) if total_items > 0 else 0
                            st.metric("% прошедших фильтр", f"{filter_ratio:.1f}%")
            
            # Рекомендации
            if st.session_state.recommender.user_item_matrix is not None:
                st.markdown("---")
                st.header("📊 Рекомендации")
                
                tab1, tab2 = st.tabs(["Для одного магазина", "Для всех магазинов"])
                
                with tab1:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        selected_shop = st.selectbox(
                            "Выберите магазин:",
                            options=st.session_state.recommender.le_magazin.classes_
                        )
                    with col2:
                        top_k = st.slider("Количество рекомендаций:", 5, 20, 10)
                    with col3:
                        apply_filter = st.checkbox("Применить фильтр популярности", value=True)
                    
                    if st.button("Получить рекомендации"):
                        recommendations = st.session_state.recommender.get_recommendations(
                            selected_shop, top_k, apply_filter
                        )
                        
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations)
                            
                            # Переименование колонок для лучшего отображения
                            display_df = rec_df.rename(columns={
                                'rank': 'Рейтинг',
                                'item': 'Товар',
                                'score': 'Прогноз',
                                'segment': 'Сегмент',
                                'model': 'Модель',
                                'avg_price': 'Средняя цена',
                                'total_qty': 'Общее кол-во',
                                'transactions': 'Транзакций',
                                'stores': 'Магазинов',
                                'popularity_score': 'Популярность'
                            })
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Статистика
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                avg_score = rec_df['score'].mean()
                                st.metric("Средний прогноз", f"{avg_score:.3f}")
                            with col2:
                                top_segment = rec_df['segment'].mode().iloc[0] if len(rec_df) > 0 else "N/A"
                                st.metric("Топ сегмент", top_segment)
                            with col3:
                                avg_price = rec_df['avg_price'].mean()
                                st.metric("Средняя цена", f"{avg_price:.2f}")
                            with col4:
                                avg_transactions = rec_df['transactions'].mean()
                                st.metric("Среднее кол-во транзакций", f"{avg_transactions:.1f}")
                        else:
                            st.info("Нет рекомендаций для данного магазина")
                
                with tab2:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        batch_top_k = st.slider("Рекомендаций на магазин:", 5, 15, 10)
                    with col2:
                        show_top_n = st.slider("Показать топ для отчета:", 3, 10, 5)
                    with col3:
                        apply_batch_filter = st.checkbox("Применить фильтр популярности", value=True, key="batch_filter")
                    
                    if st.button("Сгенерировать рекомендации для всех"):
                        with st.spinner("Генерация рекомендаций..."):
                            all_recs = st.session_state.recommender.get_all_recommendations(
                                batch_top_k, apply_batch_filter
                            )
                        
                        if all_recs:
                            # Создание сводной таблицы
                            summary_data = []
                            for shop, recs in all_recs.items():
                                for rec in recs[:show_top_n]:
                                    summary_data.append({
                                        'Магазин': shop,
                                        'Ранг': rec['rank'],
                                        'Товар': rec['item'],
                                        'Прогноз': rec['score'],
                                        'Сегмент': rec['segment'],
                                        'Модель': rec['model'],
                                        'Цена': rec['avg_price'],
                                        'Количество': rec['total_qty'],
                                        'Транзакций': rec['transactions'],
                                        'Магазинов': rec['stores'],
                                        'Популярность': rec['popularity_score']
                                    })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Статистика
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Всего рекомендаций", len(summary_data))
                            with col2:
                                avg_score = summary_df['Прогноз'].mean()
                                st.metric("Средний прогноз", f"{avg_score:.3f}")
                            with col3:
                                unique_items = summary_df['Товар'].nunique()
                                st.metric("Уникальных товаров", unique_items)
                            with col4:
                                avg_transactions = summary_df['Транзакций'].mean()
                                st.metric("Среднее кол-во транзакций", f"{avg_transactions:.1f}")
                            
                            # Анализ качества рекомендаций
                            st.subheader("🎯 Анализ качества рекомендаций")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Распределение по количеству транзакций
                                st.write("**Распределение по количеству транзакций:**")
                                transaction_bins = [1, 2, 5, 10, 20, float('inf')]
                                transaction_labels = ['1', '2-4', '5-9', '10-19', '20+']
                                summary_df['Транзакции_группа'] = pd.cut(
                                    summary_df['Транзакций'], 
                                    bins=transaction_bins, 
                                    labels=transaction_labels, 
                                    right=False
                                )
                                transaction_dist = summary_df['Транзакции_группа'].value_counts().sort_index()
                                st.bar_chart(transaction_dist)
                            
                            with col2:
                                # Топ сегменты в рекомендациях
                                st.write("**Топ сегменты в рекомендациях:**")
                                segment_counts = summary_df['Сегмент'].value_counts().head(10)
                                st.bar_chart(segment_counts)
                            
                            # Предупреждения о качестве
                            low_quality_items = summary_df[summary_df['Транзакций'] < min_sales_filter]
                            if len(low_quality_items) > 0 and apply_batch_filter:
                                st.warning(f"⚠️ Обнаружено {len(low_quality_items)} рекомендаций с количеством транзакций ниже порога ({min_sales_filter}). Проверьте настройки фильтра.")
                            
                            high_quality_items = summary_df[summary_df['Транзакций'] >= min_sales_filter * 2]
                            if len(high_quality_items) > 0:
                                quality_ratio = len(high_quality_items) / len(summary_data) * 100
                                st.success(f"✅ {quality_ratio:.1f}% рекомендаций имеют высокое качество (>{min_sales_filter*2} транзакций)")
                            
                            # Скачивание результатов
                            @st.cache_data
                            def convert_df(df):
                                return df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                            
                            csv = convert_df(summary_df)
                            st.download_button(
                                label="📥 Скачать рекомендации (CSV)",
                                data=csv,
                                file_name=f'recommendations_min{min_sales_filter}_transactions.csv',
                                mime='text/csv'
                            )
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            st.error("Проверьте формат данных и попробуйте снова.")
    
    else:
        st.info("👆 Загрузите Excel файл для начала работы")
        
        # Пример структуры данных
        st.markdown("### 📋 Требуемые колонки:")
        st.markdown("- **Magazin** - название магазина")
        st.markdown("- **Art** - код/название товара") 
        st.markdown("- **Segment** - сегмент товара")
        st.markdown("- **Model** - модель товара")
        st.markdown("- **Price** - цена")
        st.markdown("- **Qty** - количество")
        
        st.markdown("### 🎯 Новые возможности:")
        st.markdown("- **Фильтрация по популярности** - исключение товаров с малым количеством продаж")
        st.markdown("- **Анализ качества рекомендаций** - статистика по транзакциям и магазинам")
        st.markdown("- **Гибкие настройки** - возможность установить минимальный порог транзакций")
        st.markdown("- **Детальная статистика** - информация о популярности каждого товара")

if __name__ == "__main__":
    create_dashboard()
