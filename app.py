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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Класс для обработки и предобработки данных"""
    
    def __init__(self):
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df):
        """Предобработка данных с расширенными признаками"""
        df = df.copy()
        df['Datasales'] = pd.to_datetime(df['Datasales'])
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        
        # Временные признаки
        df['Month'] = df['Datasales'].dt.month
        df['Quarter'] = df['Datasales'].dt.quarter
        df['Weekday'] = df['Datasales'].dt.dayofweek
        df['DayOfMonth'] = df['Datasales'].dt.day
        df['WeekOfYear'] = df['Datasales'].dt.isocalendar().week
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
        
        # Бизнес-метрики
        df['Revenue'] = df['Price'] * df['Qty']
        df['PriceCategory'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['QtyCategory'] = pd.cut(df['Qty'], bins=3, labels=['Low', 'Medium', 'High'])
        
        # Энкодинг
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'])
        df['art_encoded'] = self.le_art.fit_transform(df['Art'])
        
        return df
    
    def create_aggregated_data(self, df):
        """Создание агрегированных данных"""
        agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg({
            'Qty': ['sum', 'mean', 'count', 'std'],
            'Revenue': ['sum', 'mean', 'std'],
            'Price': ['mean', 'min', 'max', 'std'],
            'Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'IsWeekend': 'mean',
            'Segment': 'first',
            'Model': 'first',
            'Describe': 'first'
        }).reset_index()
        
        # Упрощение колонок
        agg_data.columns = ['magazin_encoded', 'art_encoded', 'Magazin', 'Art', 
                           'qty_sum', 'qty_mean', 'freq', 'qty_std',
                           'revenue_sum', 'revenue_mean', 'revenue_std',
                           'price_mean', 'price_min', 'price_max', 'price_std',
                           'peak_month', 'weekend_ratio', 'Segment', 'Model', 'Describe']
        
        # Заполнение NaN
        agg_data = agg_data.fillna(0)
        
        # Расчет рейтинга с учетом нескольких факторов
        agg_data['rating'] = (
            np.log1p(agg_data['qty_sum']) * 0.3 +
            np.log1p(agg_data['revenue_sum']) * 0.3 +
            np.log1p(agg_data['freq']) * 0.2 +
            (agg_data['weekend_ratio'] * 0.1) +
            (1 / (1 + agg_data['price_std'] + 1e-6)) * 0.1
        )
        
        # Нормализация рейтинга
        agg_data['rating'] = ((agg_data['rating'] - agg_data['rating'].min()) / 
                             (agg_data['rating'].max() - agg_data['rating'].min()) * 4 + 1)
        
        return agg_data

class RecommenderModels:
    """Класс для моделей рекомендаций"""
    
    def __init__(self):
        self.svd_model = None
        self.nmf_model = None
        self.rf_model = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.weights = {'svd': 0.4, 'nmf': 0.3, 'similarity': 0.2, 'content': 0.1}
    
    def create_user_item_matrix(self, df):
        """Создание матрицы пользователь-товар"""
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        user_item_matrix = csr_matrix((df['rating'], 
                                     (df['magazin_encoded'], df['art_encoded'])), 
                                    shape=(n_users, n_items))
        
        self.user_item_matrix = user_item_matrix.toarray()
        return self.user_item_matrix
    
    def train_models(self, user_item_matrix, content_features=None):
        """Обучение всех моделей"""
        # SVD
        n_components_svd = min(50, min(user_item_matrix.shape) - 1)
        if n_components_svd > 0:
            self.svd_model = TruncatedSVD(n_components=n_components_svd, random_state=42)
            self.svd_model.fit(user_item_matrix)
        
        # NMF
        n_components_nmf = min(30, min(user_item_matrix.shape) - 1)
        if n_components_nmf > 0:
            self.nmf_model = NMF(n_components=n_components_nmf, random_state=42, max_iter=500)
            self.nmf_model.fit(user_item_matrix)
        
        # Item similarity
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        
        # Content-based RF
        if content_features is not None and len(content_features) > 0:
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    
    def predict_single_rating(self, user_id, item_id):
        """Предсказание одного рейтинга ансамблем"""
        predictions = []
        
        # SVD
        if self.svd_model and user_id < self.user_item_matrix.shape[0]:
            try:
                user_factors = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('svd', svd_pred))
            except:
                pass
        
        # NMF
        if self.nmf_model and user_id < self.user_item_matrix.shape[0]:
            try:
                user_factors = self.nmf_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.nmf_model.components_[:, item_id]
                nmf_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('nmf', nmf_pred))
            except:
                pass
        
        # Item similarity
        if (user_id < self.user_item_matrix.shape[0] and 
            item_id < self.user_item_matrix.shape[1]):
            try:
                user_ratings = self.user_item_matrix[user_id]
                similar_items = self.item_similarity[item_id]
                
                numerator = np.sum(similar_items * user_ratings)
                denominator = np.sum(np.abs(similar_items))
                
                if denominator > 0:
                    similarity_pred = numerator / denominator
                    predictions.append(('similarity', similarity_pred))
            except:
                pass
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights.get(method, 0.25) for method, pred in predictions)
            total_weight = sum(self.weights.get(method, 0.25) for method, _ in predictions)
            return weighted_sum / total_weight if total_weight > 0 else 2.5
        
        return 2.5

class EnsembleRecommenderSystem:
    """Главный класс рекомендательной системы"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = RecommenderModels()
        self.processed_data = None
        self.raw_data = None
    
    def build_ensemble_model(self, df, test_size=0.2):
        """Построение ансамбля моделей"""
        self.raw_data = df.copy()
        
        # Предобработка
        processed_df = self.data_processor.preprocess_data(df)
        self.processed_data = self.data_processor.create_aggregated_data(processed_df)
        
        # Создание матрицы
        user_item_matrix = self.models.create_user_item_matrix(self.processed_data)
        
        # Обучение моделей
        self.models.train_models(user_item_matrix)
        
        # Разделение данных для оценки
        train_data, test_data = train_test_split(self.processed_data, test_size=test_size, random_state=42)
        
        # Метрики
        train_predictions = self._predict_ratings_for_evaluation(train_data)
        test_predictions = self._predict_ratings_for_evaluation(test_data)
        
        train_rmse = np.sqrt(np.mean((train_data['rating'] - train_predictions) ** 2))
        test_rmse = np.sqrt(np.mean((test_data['rating'] - test_predictions) ** 2))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'n_users': len(self.processed_data['magazin_encoded'].unique()),
            'n_items': len(self.processed_data['art_encoded'].unique()),
            'sparsity': 1 - np.count_nonzero(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        }
    
    def _predict_ratings_for_evaluation(self, test_data):
        """Предсказание рейтингов для оценки"""
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.models.predict_single_rating(row['magazin_encoded'], row['art_encoded'])
            predictions.append(pred)
        return np.array(predictions)
    
    def get_recommendations(self, magazin_name, top_k=10, filters=None):
        """Получение персонализированных рекомендаций"""
        if self.models.user_item_matrix is None:
            return None
        
        try:
            user_id = self.data_processor.le_magazin.transform([magazin_name])[0]
        except:
            return None
        
        if user_id >= self.models.user_item_matrix.shape[0]:
            return None
        
        # Фильтрация данных
        filtered_data = self.processed_data.copy()
        if filters:
            if 'segments' in filters and filters['segments']:
                filtered_data = filtered_data[filtered_data['Segment'].isin(filters['segments'])]
            if 'price_range' in filters and filters['price_range']:
                min_price, max_price = filters['price_range']
                filtered_data = filtered_data[
                    (filtered_data['price_mean'] >= min_price) & 
                    (filtered_data['price_mean'] <= max_price)
                ]
            if 'models' in filters and filters['models']:
                filtered_data = filtered_data[filtered_data['Model'].isin(filters['models'])]
        
        # Получение рекомендаций
        user_ratings = self.models.user_item_matrix[user_id]
        predictions = []
        
        for _, row in filtered_data.iterrows():
            item_id = row['art_encoded']
            if user_ratings[item_id] == 0:  # Только неоцененные товары
                pred_rating = self.models.predict_single_rating(user_id, item_id)
                predictions.append((row, pred_rating))
        
        # Сортировка и топ-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = predictions[:top_k]
        
        # Формирование результатов
        recommendations = []
        for rank, (item_info, score) in enumerate(top_items, 1):
            rec = {
                'rank': rank,
                'item': item_info['Art'],
                'score': score,
                'segment': item_info['Segment'],
                'model': item_info['Model'],
                'describe': item_info['Describe'],
                'avg_price': item_info['price_mean'],
                'expected_qty': item_info['qty_mean'],
                'frequency': item_info['freq'],
                'revenue_potential': item_info['revenue_mean']
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_top_products(self, filters=None, top_k=20):
        """Получение топ товаров по общей популярности"""
        if self.processed_data is None:
            return None
        
        data = self.processed_data.copy()
        
        # Применение фильтров
        if filters:
            if 'segments' in filters and filters['segments']:
                data = data[data['Segment'].isin(filters['segments'])]
            if 'price_range' in filters and filters['price_range']:
                min_price, max_price = filters['price_range']
                data = data[(data['price_mean'] >= min_price) & (data['price_mean'] <= max_price)]
            if 'models' in filters and filters['models']:
                data = data[data['Model'].isin(filters['models'])]
        
        # Сортировка по рейтингу
        top_products = data.nlargest(top_k, 'rating')
        
        results = []
        for rank, (_, product) in enumerate(top_products.iterrows(), 1):
            results.append({
                'rank': rank,
                'item': product['Art'],
                'rating': product['rating'],
                'segment': product['Segment'],
                'model': product['Model'],
                'describe': product['Describe'],
                'avg_price': product['price_mean'],
                'total_qty': product['qty_sum'],
                'total_revenue': product['revenue_sum'],
                'frequency': product['freq']
            })
        
        return results
    
    def get_analytics_data(self):
        """Получение данных для аналитики"""
        if self.processed_data is None:
            return None
        
        return {
            'processed_data': self.processed_data,
            'raw_data': self.raw_data,
            'segments': sorted(self.processed_data['Segment'].unique()),
            'models': sorted(self.processed_data['Model'].unique()),
            'shops': sorted(self.data_processor.le_magazin.classes_),
            'price_range': (
                self.processed_data['price_mean'].min(),
                self.processed_data['price_mean'].max()
            )
        }

def create_filters_sidebar(analytics_data):
    """Создание боковой панели с фильтрами"""
    st.sidebar.header("🎛️ Фильтры")
    
    filters = {}
    
    # Фильтр по сегментам
    segments = st.sidebar.multiselect(
        "Сегменты:",
        options=analytics_data['segments'],
        default=[]
    )
    if segments:
        filters['segments'] = segments
    
    # Фильтр по моделям
    models = st.sidebar.multiselect(
        "Модели:",
        options=analytics_data['models'],
        default=[]
    )
    if models:
        filters['models'] = models
    
    # Фильтр по цене
    price_min, price_max = analytics_data['price_range']
    price_range = st.sidebar.slider(
        "Диапазон цен:",
        min_value=float(price_min),
        max_value=float(price_max),
        value=(float(price_min), float(price_max)),
        step=1.0
    )
    if price_range != (price_min, price_max):
        filters['price_range'] = price_range
    
    return filters

def create_dashboard():
    st.set_page_config(page_title="Рекомендательная система", layout="wide")
    
    st.title("🛍️ Усовершенствованная рекомендательная система")
    st.markdown("*Ансамбль алгоритмов с расширенной функциональностью и фильтрацией*")
    st.markdown("---")
    
    # Инициализация
    if 'recommender' not in st.session_state:
        st.session_state.recommender = EnsembleRecommenderSystem()
    
    # Загрузка файла
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Проверка колонок
            required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
                return
            
            # Основная информация
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
                with st.spinner("Обучение ансамбля моделей..."):
                    metrics = st.session_state.recommender.build_ensemble_model(df)
                
                st.success("Модель готова!")
                
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
            
            # Основной интерфейс
            if st.session_state.recommender.processed_data is not None:
                analytics_data = st.session_state.recommender.get_analytics_data()
                filters = create_filters_sidebar(analytics_data)
                
                st.markdown("---")
                
                # Измененный порядок вкладок: 1.Рекомендации 2.ТОП товары 3.Аналитика
                tab1, tab2, tab3 = st.tabs(["🎯 Рекомендации", "🏆 ТОП товары", "📊 Аналитика"])
                
                with tab1:
                    st.header("Персональные рекомендации")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        selected_shop = st.selectbox(
                            "Выберите магазин:",
                            options=analytics_data['shops']
                        )
                    with col2:
                        top_k = st.slider("Количество рекомендаций:", 5, 20, 10)
                    
                    if st.button("Получить рекомендации", type="primary"):
                        recommendations = st.session_state.recommender.get_recommendations(
                            selected_shop, top_k, filters
                        )
                        
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations)
                            
                            # Отображение таблицы
                            display_df = rec_df.copy()
                            display_df['score'] = display_df['score'].round(3)
                            display_df['avg_price'] = display_df['avg_price'].round(2)
                            display_df['expected_qty'] = display_df['expected_qty'].round(1)
                            display_df['revenue_potential'] = display_df['revenue_potential'].round(2)
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # График рекомендаций
                            fig = px.bar(
                                rec_df.head(10), x='item', y='score',
                                title=f"Рекомендации для {selected_shop}",
                                color='score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Скачивание
                            csv = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Скачать", csv, f"recommendations_{selected_shop}.csv", "text/csv")
                        else:
                            st.warning("Рекомендации не найдены")
                
                with tab2:
                    st.header("Топ товары по популярности")
                    
                    top_k_products = st.slider("Количество товаров:", 10, 50, 20)
                    
                    if st.button("Показать ТОП товары", type="primary"):
                        top_products = st.session_state.recommender.get_top_products(filters, top_k_products)
                        
                        if top_products:
                            top_df = pd.DataFrame(top_products)
                            
                            # Форматирование
                            display_df = top_df.copy()
                            display_df['rating'] = display_df['rating'].round(3)
                            display_df['avg_price'] = display_df['avg_price'].round(2)
                            display_df['total_revenue'] = display_df['total_revenue'].round(2)
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # График топ товаров
                            fig = px.bar(
                                top_df.head(15), x='item', y='rating',
                                title="Топ товары по рейтингу",
                                color='rating',
                                color_continuous_scale='plasma'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Скачивание
                            csv = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Скачать ТОП", csv, "top_products.csv", "text/csv")
                
                with tab3:
                    st.header("Аналитика данных")
                    
                    data = analytics_data['processed_data']
                    
                    # Применение фильтров к аналитике
                    filtered_data = data.copy()
                    if filters:
                        if 'segments' in filters:
                            filtered_data = filtered_data[filtered_data['Segment'].isin(filters['segments'])]
                        if 'price_range' in filters:
                            min_p, max_p = filters['price_range']
                            filtered_data = filtered_data[
                                (filtered_data['price_mean'] >= min_p) & 
                                (filtered_data['price_mean'] <= max_p)
                            ]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Распределение рейтингов
                        fig1 = px.histogram(
                            filtered_data, x='rating', bins=20,
                            title="Распределение рейтингов"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Средний рейтинг по сегментам
                        if len(filtered_data) > 0:
                            segment_stats = filtered_data.groupby('Segment').agg({
                                'rating': 'mean',
                                'qty_sum': 'sum',
                                'revenue_sum': 'sum'
                            }).reset_index()
                            
                            fig2 = px.bar(
                                segment_stats, x='Segment', y='rating',
                                title="Средний рейтинг по сегментам"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    # Корреляционная матрица
                    if len(filtered_data) > 0:
                        numeric_cols = ['qty_sum', 'revenue_sum', 'price_mean', 'freq', 'rating']
                        corr_matrix = filtered_data[numeric_cols].corr()
                        
                        fig3 = px.imshow(
                            corr_matrix, 
                            title="Корреляция метрик",
                            aspect="auto",
                            color_continuous_scale='RdBu'
                        )
                        st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    else:
        st.info("Загрузите Excel файл для начала работы")
        
        # Пример данных
        st.markdown("### Пример структуры данных:")
        example_data = {
            'Magazin': ['Shop_A', 'Shop_B', 'Shop_A', 'Shop_C'],
            'Datasales': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
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

if __name__ == "__main__":
    create_dashboard()
