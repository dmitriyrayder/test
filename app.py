import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
    
    def preprocess_data(self, df):
        df = df.copy()
        
        # Преобразуем все значения в строки для избежания ошибки с mixed types
        df['Magazin'] = df['Magazin'].astype(str)
        df['Art'] = df['Art'].astype(str)
        
        # Исправление парсинга даты - используем dayfirst=True для DD.MM.YYYY
        df['Datasales'] = pd.to_datetime(df['Datasales'], dayfirst=True, errors='coerce')
        
        # Удаляем строки с NaN значениями в критических колонках
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty', 'Datasales'])
        
        # Убираем строки с 'nan' в строковом формате
        df = df[df['Magazin'] != 'nan']
        df = df[df['Art'] != 'nan']
        
        # Проверяем, что у нас есть данные после очистки
        if len(df) == 0:
            raise ValueError("После очистки данных не осталось записей")
        
        # Временные признаки
        df['Month'] = df['Datasales'].dt.month
        df['Weekday'] = df['Datasales'].dt.dayofweek
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
        
        # Бизнес-метрики
        df['Revenue'] = df['Price'] * df['Qty']
        
        # Энкодинг - теперь все значения являются строками
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'])
        df['art_encoded'] = self.le_art.fit_transform(df['Art'])
        
        return df
    
    def create_aggregated_data(self, df):
        agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg({
            'Qty': ['sum', 'mean', 'count'],
            'Revenue': ['sum', 'mean'],
            'Price': ['mean', 'std'],
            'IsWeekend': 'mean',
            'Segment': 'first',
            'Model': 'first',
            'Describe': 'first'
        }).reset_index()
        
        agg_data.columns = ['magazin_encoded', 'art_encoded', 'Magazin', 'Art', 
                           'qty_sum', 'qty_mean', 'freq', 'revenue_sum', 'revenue_mean',
                           'price_mean', 'price_std', 'weekend_ratio', 'Segment', 'Model', 'Describe']
        
        agg_data = agg_data.fillna(0)
        
        # Расчет рейтинга
        agg_data['rating'] = (
            np.log1p(agg_data['qty_sum']) * 0.4 +
            np.log1p(agg_data['revenue_sum']) * 0.4 +
            np.log1p(agg_data['freq']) * 0.2
        )
        
        # Нормализация рейтинга
        rating_range = agg_data['rating'].max() - agg_data['rating'].min()
        if rating_range > 0:
            agg_data['rating'] = ((agg_data['rating'] - agg_data['rating'].min()) / rating_range * 4 + 1)
        else:
            agg_data['rating'] = 2.5
        
        return agg_data

class RecommenderModels:
    def __init__(self):
        self.svd_model = None
        self.nmf_model = None
        self.item_similarity = None
        self.user_item_matrix = None
    
    def create_user_item_matrix(self, df):
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        user_item_matrix = csr_matrix((df['rating'], 
                                     (df['magazin_encoded'], df['art_encoded'])), 
                                    shape=(n_users, n_items))
        
        self.user_item_matrix = user_item_matrix.toarray()
        return self.user_item_matrix
    
    def train_models(self, user_item_matrix):
        # Проверяем минимальные размеры для обучения
        if min(user_item_matrix.shape) < 2:
            st.warning("Недостаточно данных для обучения моделей")
            return
            
        # SVD
        n_components_svd = min(30, min(user_item_matrix.shape) - 1)
        if n_components_svd > 0:
            try:
                self.svd_model = TruncatedSVD(n_components=n_components_svd, random_state=42)
                self.svd_model.fit(user_item_matrix)
            except Exception as e:
                st.warning(f"Ошибка обучения SVD: {e}")
        
        # NMF
        n_components_nmf = min(20, min(user_item_matrix.shape) - 1)
        if n_components_nmf > 0:
            try:
                self.nmf_model = NMF(n_components=n_components_nmf, random_state=42, max_iter=200)
                self.nmf_model.fit(user_item_matrix)
            except Exception as e:
                st.warning(f"Ошибка обучения NMF: {e}")
        
        # Item similarity
        try:
            self.item_similarity = cosine_similarity(user_item_matrix.T)
        except Exception as e:
            st.warning(f"Ошибка расчета похожести: {e}")
    
    def predict_rating(self, user_id, item_id):
        predictions = []
        
        # SVD prediction
        if self.svd_model and user_id < self.user_item_matrix.shape[0] and item_id < self.user_item_matrix.shape[1]:
            try:
                user_factors = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(svd_pred)
            except:
                pass
        
        # NMF prediction
        if self.nmf_model and user_id < self.user_item_matrix.shape[0] and item_id < self.user_item_matrix.shape[1]:
            try:
                user_factors = self.nmf_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.nmf_model.components_[:, item_id]
                nmf_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(nmf_pred)
            except:
                pass
        
        # Similarity prediction
        if (self.item_similarity is not None and 
            user_id < self.user_item_matrix.shape[0] and 
            item_id < self.user_item_matrix.shape[1]):
            try:
                user_ratings = self.user_item_matrix[user_id]
                similar_items = self.item_similarity[item_id]
                
                numerator = np.sum(similar_items * user_ratings)
                denominator = np.sum(np.abs(similar_items))
                
                if denominator > 0:
                    similarity_pred = numerator / denominator
                    predictions.append(similarity_pred)
            except:
                pass
        
        return np.mean(predictions) if predictions else 2.5

class EnsembleRecommenderSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = RecommenderModels()
        self.processed_data = None
        self.raw_data = None
    
    def build_model(self, df, test_size=0.2):
        try:
            self.raw_data = df.copy()
            
            # Предобработка
            processed_df = self.data_processor.preprocess_data(df)
            self.processed_data = self.data_processor.create_aggregated_data(processed_df)
            
            # Создание матрицы и обучение
            user_item_matrix = self.models.create_user_item_matrix(self.processed_data)
            self.models.train_models(user_item_matrix)
            
            # Оценка модели
            if len(self.processed_data) > 1:
                train_data, test_data = train_test_split(self.processed_data, test_size=test_size, random_state=42)
                
                train_predictions = [self.models.predict_rating(row['magazin_encoded'], row['art_encoded']) 
                                   for _, row in train_data.iterrows()]
                test_predictions = [self.models.predict_rating(row['magazin_encoded'], row['art_encoded']) 
                                  for _, row in test_data.iterrows()]
                
                train_rmse = np.sqrt(np.mean((train_data['rating'] - train_predictions) ** 2))
                test_rmse = np.sqrt(np.mean((test_data['rating'] - test_predictions) ** 2))
            else:
                train_rmse = test_rmse = 0.0
            
            return {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'n_users': len(self.processed_data['magazin_encoded'].unique()),
                'n_items': len(self.processed_data['art_encoded'].unique()),
                'sparsity': 1 - np.count_nonzero(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
            }
            
        except Exception as e:
            st.error(f"Ошибка при построении модели: {str(e)}")
            return None
    
    def get_recommendations(self, magazin_name, top_k=10, filters=None):
        if self.models.user_item_matrix is None:
            return None
        
        try:
            # Преобразуем magazin_name в строку для консистентности
            magazin_name = str(magazin_name)
            user_id = self.data_processor.le_magazin.transform([magazin_name])[0]
        except ValueError:
            st.error(f"Магазин '{magazin_name}' не найден в данных")
            return None
        except Exception as e:
            st.error(f"Ошибка при поиске магазина: {str(e)}")
            return None
        
        if user_id >= self.models.user_item_matrix.shape[0]:
            return None
        
        # Фильтрация
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
        
        # Получение рекомендаций
        user_ratings = self.models.user_item_matrix[user_id]
        predictions = []
        
        for _, row in filtered_data.iterrows():
            item_id = row['art_encoded']
            if item_id < len(user_ratings) and user_ratings[item_id] == 0:
                pred_rating = self.models.predict_rating(user_id, item_id)
                predictions.append((row, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = predictions[:top_k]
        
        recommendations = []
        for rank, (item_info, score) in enumerate(top_items, 1):
            recommendations.append({
                'rank': rank,
                'item': item_info['Art'],
                'score': round(score, 3),
                'segment': item_info['Segment'],
                'model': item_info['Model'],
                'describe': item_info['Describe'],
                'avg_price': round(item_info['price_mean'], 2),
                'expected_qty': round(item_info['qty_mean'], 1),
                'frequency': item_info['freq'],
                'revenue_potential': round(item_info['revenue_mean'], 2)
            })
        
        return recommendations
    
    def get_top_products(self, filters=None, top_k=20):
        if self.processed_data is None:
            return None
        
        data = self.processed_data.copy()
        
        if filters:
            if 'segments' in filters and filters['segments']:
                data = data[data['Segment'].isin(filters['segments'])]
            if 'price_range' in filters and filters['price_range']:
                min_price, max_price = filters['price_range']
                data = data[(data['price_mean'] >= min_price) & (data['price_mean'] <= max_price)]
        
        top_products = data.nlargest(top_k, 'rating')
        
        results = []
        for rank, (_, product) in enumerate(top_products.iterrows(), 1):
            results.append({
                'rank': rank,
                'item': product['Art'],
                'rating': round(product['rating'], 3),
                'segment': product['Segment'],
                'model': product['Model'],
                'describe': product['Describe'],
                'avg_price': round(product['price_mean'], 2),
                'total_qty': product['qty_sum'],
                'total_revenue': round(product['revenue_sum'], 2),
                'frequency': product['freq']
            })
        
        return results
    
    def get_analytics_data(self):
        if self.processed_data is None:
            return None
        
        return {
            'processed_data': self.processed_data,
            'segments': sorted(self.processed_data['Segment'].unique()),
            'models': sorted(self.processed_data['Model'].unique()),
            'shops': sorted(self.data_processor.le_magazin.classes_),
            'price_range': (
                self.processed_data['price_mean'].min(),
                self.processed_data['price_mean'].max()
            )
        }

def create_filters_sidebar(analytics_data):
    st.sidebar.header("🎛️ Фильтры")
    filters = {}
    
    segments = st.sidebar.multiselect("Сегменты:", options=analytics_data['segments'], default=[])
    if segments:
        filters['segments'] = segments
    
    models = st.sidebar.multiselect("Модели:", options=analytics_data['models'], default=[])
    if models:
        filters['models'] = models
    
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
    st.title("🛍️ Рекомендательная система")
    st.markdown("*Ансамбль алгоритмов с фильтрацией*")
    st.markdown("---")
    
    if 'recommender' not in st.session_state:
        st.session_state.recommender = EnsembleRecommenderSystem()
    
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
                return
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Записей", len(df))
            with col2:
                st.metric("Магазинов", df['Magazin'].nunique())
            with col3:
                st.metric("Товаров", df['Art'].nunique())
            with col4:
                st.metric("Сегментов", df['Segment'].nunique())
            
            if st.sidebar.button("🚀 Построить модель", type="primary"):
                with st.spinner("Обучение модели..."):
                    metrics = st.session_state.recommender.build_model(df)
                
                if metrics:
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
            
            if st.session_state.recommender.processed_data is not None:
                analytics_data = st.session_state.recommender.get_analytics_data()
                filters = create_filters_sidebar(analytics_data)
                
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["🎯 Рекомендации", "🏆 ТОП товары", "📊 Аналитика"])
                
                with tab1:
                    st.header("Персональные рекомендации")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        selected_shop = st.selectbox("Выберите магазин:", options=analytics_data['shops'])
                    with col2:
                        top_k = st.slider("Количество:", 5, 20, 10)
                    
                    if st.button("Получить рекомендации", type="primary"):
                        recommendations = st.session_state.recommender.get_recommendations(
                            selected_shop, top_k, filters
                        )
                        
                        if recommendations:
                            rec_df = pd.DataFrame(recommendations)
                            st.dataframe(rec_df, use_container_width=True)
                            
                            fig = px.bar(
                                rec_df.head(10), x='item', y='score',
                                title=f"Рекомендации для {selected_shop}",
                                color='score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            csv = rec_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Скачать", csv, f"recommendations_{selected_shop}.csv", "text/csv")
                        else:
                            st.warning("Рекомендации не найдены")
                
                with tab2:
                    st.header("Топ товары")
                    
                    top_k_products = st.slider("Количество товаров:", 10, 50, 20)
                    
                    if st.button("Показать ТОП", type="primary"):
                        top_products = st.session_state.recommender.get_top_products(filters, top_k_products)
                        
                        if top_products:
                            top_df = pd.DataFrame(top_products)
                            st.dataframe(top_df, use_container_width=True)
                            
                            fig = px.bar(
                                top_df.head(15), x='item', y='rating',
                                title="Топ товары по рейтингу",
                                color='rating',
                                color_continuous_scale='plasma'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            csv = top_df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Скачать ТОП", csv, "top_products.csv", "text/csv")
                
                with tab3:
                    st.header("Аналитика")
                    
                    data = analytics_data['processed_data']
                    
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
                        fig1 = px.histogram(filtered_data, x='rating', nbins=20, title="Распределение рейтингов")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        if len(filtered_data) > 0:
                            segment_stats = filtered_data.groupby('Segment')['rating'].mean().reset_index()
                            fig2 = px.bar(segment_stats, x='Segment', y='rating', title="Рейтинг по сегментам")
                            st.plotly_chart(fig2, use_container_width=True)
                    
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
            st.error("Проверьте структуру данных и убедитесь, что все необходимые колонки присутствуют")
    
    else:
        st.info("Загрузите Excel файл для начала работы")
        
        st.markdown("### Пример структуры данных:")
        example_data = {
            'Magazin': ['Shop_A', 'Shop_B', 'Shop_A', 'Shop_C'],
            'Datasales': ['15.01.2024', '16.01.2024', '17.01.2024', '18.01.2024'],
            'Art': ['Item_001', 'Item_002', 'Item_003', 'Item_001'],
            'Describe': ['Описание 1', 'Описание 2', 'Описание 3', 'Описание 1'],
            'Model': ['Model_X', 'Model_Y', 'Model_Z', 'Model_X'],
            'Segment': ['Electronics', 'Clothing', 'Electronics', 'Electronics'],
            'Price': [100, 50, 150, 105],
            'Qty': [2, 1, 3, 1]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)

if __name__ == "__main__":
    create_dashboard()
