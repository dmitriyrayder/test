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
warnings.filterwarnings('ignore')

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
        
    def preprocess_data(self, df):
        """Предобработка данных"""
        df = df.copy()
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        
        # Создание временных признаков
        df['Month'] = df['Datasales'].dt.month
        df['Quarter'] = df['Datasales'].dt.quarter
        df['Weekday'] = df['Datasales'].dt.dayofweek
        df['DayOfMonth'] = df['Datasales'].dt.day
        
        # Бизнес-метрики
        df['Revenue'] = df['Price'] * df['Qty']
        df['PriceCategory'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Энкодинг
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'])
        df['art_encoded'] = self.le_art.fit_transform(df['Art'])
        
        # Агрегация по магазин-товар
        agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg({
            'Qty': ['sum', 'mean', 'count'],
            'Revenue': ['sum', 'mean'],
            'Price': ['mean', 'min', 'max'],
            'Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'Segment': 'first',
            'Model': 'first',
            'Describe': 'first'
        }).reset_index()
        
        # Упрощение колонок
        agg_data.columns = ['magazin_encoded', 'art_encoded', 'Magazin', 'Art', 
                           'qty_sum', 'qty_mean', 'freq', 'revenue_sum', 'revenue_mean',
                           'price_mean', 'price_min', 'price_max', 'peak_month',
                           'Segment', 'Model', 'Describe']
        
        # Создание рейтинга
        agg_data['rating'] = (
            np.log1p(agg_data['qty_sum']) * 0.4 +
            np.log1p(agg_data['revenue_sum']) * 0.4 +
            np.log1p(agg_data['freq']) * 0.2
        )
        
        # Нормализация рейтинга
        agg_data['rating'] = (agg_data['rating'] - agg_data['rating'].min()) / (agg_data['rating'].max() - agg_data['rating'].min()) * 4 + 1
        
        self.processed_data = agg_data
        return agg_data
    
    def create_user_item_matrix(self, df):
        """Создание матрицы пользователь-товар"""
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        # Создание разреженной матрицы
        user_item_matrix = csr_matrix((df['rating'], 
                                     (df['magazin_encoded'], df['art_encoded'])), 
                                    shape=(n_users, n_items))
        
        self.user_item_matrix = user_item_matrix.toarray()
        return self.user_item_matrix
    
    def prepare_content_features(self, df):
        """Подготовка контентных признаков"""
        # Создание признаков товаров
        item_features = df.groupby('art_encoded').agg({
            'price_mean': 'first',
            'Segment': 'first',
            'Model': 'first',
            'qty_mean': 'first',
            'revenue_mean': 'first'
        }).reset_index()
        
        # One-hot encoding для категориальных признаков
        segment_dummies = pd.get_dummies(item_features['Segment'], prefix='segment')
        model_dummies = pd.get_dummies(item_features['Model'], prefix='model')
        
        # Объединение признаков
        features = pd.concat([
            item_features[['art_encoded', 'price_mean', 'qty_mean', 'revenue_mean']],
            segment_dummies,
            model_dummies
        ], axis=1)
        
        self.feature_columns = [col for col in features.columns if col != 'art_encoded']
        
        # Нормализация числовых признаков
        numeric_cols = ['price_mean', 'qty_mean', 'revenue_mean']
        features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        
        return features
    
    def build_ensemble_model(self, df, test_size=0.2):
        """Построение ансамбля моделей"""
        # Предобработка
        df = self.preprocess_data(df)
        user_item_matrix = self.create_user_item_matrix(df)
        content_features = self.prepare_content_features(df)
        
        # Разделение данных
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # 1. SVD (Matrix Factorization)
        self.svd_model = TruncatedSVD(n_components=min(50, min(user_item_matrix.shape)), random_state=42)
        svd_matrix = self.svd_model.fit_transform(user_item_matrix)
        
        # 2. NMF (Non-negative Matrix Factorization)
        self.nmf_model = NMF(n_components=min(30, min(user_item_matrix.shape)), random_state=42, max_iter=500)
        nmf_matrix = self.nmf_model.fit_transform(user_item_matrix)
        
        # 3. Item-based Collaborative Filtering
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        
        # 4. Content-based Random Forest
        if len(content_features) > 0:
            # Подготовка данных для RF
            rf_data = df.merge(content_features, on='art_encoded', how='left')
            X_rf = rf_data[self.feature_columns].fillna(0)
            y_rf = rf_data['rating']
            
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            self.rf_model.fit(X_rf, y_rf)
        
        # Вычисление метрик
        train_predictions = self.predict_ratings_for_evaluation(train_data)
        test_predictions = self.predict_ratings_for_evaluation(test_data)
        
        train_rmse = np.sqrt(np.mean((train_data['rating'] - train_predictions) ** 2))
        test_rmse = np.sqrt(np.mean((test_data['rating'] - test_predictions) ** 2))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'n_users': len(df['magazin_encoded'].unique()),
            'n_items': len(df['art_encoded'].unique()),
            'sparsity': 1 - np.count_nonzero(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        }
    
    def predict_ratings_for_evaluation(self, test_data):
        """Предсказание рейтингов для оценки модели"""
        predictions = []
        
        for _, row in test_data.iterrows():
            user_id = row['magazin_encoded']
            item_id = row['art_encoded']
            
            pred = self.predict_single_rating(user_id, item_id)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_single_rating(self, user_id, item_id):
        """Предсказание одного рейтинга"""
        predictions = []
        
        # SVD предсказание
        if self.svd_model and user_id < self.user_item_matrix.shape[0]:
            user_factors = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
            item_factors = self.svd_model.components_[:, item_id]
            svd_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
            predictions.append(('svd', svd_pred))
        
        # NMF предсказание
        if self.nmf_model and user_id < self.user_item_matrix.shape[0]:
            user_factors = self.nmf_model.transform(self.user_item_matrix[user_id:user_id+1])
            item_factors = self.nmf_model.components_[:, item_id]
            nmf_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
            predictions.append(('nmf', nmf_pred))
        
        # Item similarity предсказание
        if user_id < self.user_item_matrix.shape[0] and item_id < self.user_item_matrix.shape[1]:
            user_ratings = self.user_item_matrix[user_id]
            similar_items = self.item_similarity[item_id]
            
            # Взвешенное среднее по похожим товарам
            numerator = np.sum(similar_items * user_ratings)
            denominator = np.sum(np.abs(similar_items))
            
            if denominator > 0:
                similarity_pred = numerator / denominator
                predictions.append(('similarity', similarity_pred))
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights.get(method, 0.25) for method, pred in predictions)
            total_weight = sum(self.weights.get(method, 0.25) for method, _ in predictions)
            return weighted_sum / total_weight if total_weight > 0 else 2.5
        
        return 2.5  # Средний рейтинг по умолчанию
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Получение рекомендаций для магазина"""
        if self.user_item_matrix is None:
            return None
        
        try:
            user_id = self.le_magazin.transform([magazin_name])[0]
        except:
            return None
        
        if user_id >= self.user_item_matrix.shape[0]:
            return None
        
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
                        'segment': info['Segment'],
                        'model': info['Model'],
                        'avg_price': info['price_mean'],
                        'expected_qty': info['qty_mean']
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
            except:
                continue
        
        return recommendations
    
    def get_all_recommendations(self, top_k=10):
        """Получение рекомендаций для всех магазинов"""
        if self.user_item_matrix is None:
            return None
        
        all_recommendations = {}
        for magazin_name in self.le_magazin.classes_:
            recommendations = self.get_recommendations(magazin_name, top_k)
            if recommendations:
                all_recommendations[magazin_name] = recommendations
        
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
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Проверка колонок
            required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
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
                with st.spinner("Обучение ансамбля моделей..."):
                    metrics = st.session_state.recommender.build_ensemble_model(df)
                
                st.success("Ансамбль моделей обучен!")
                
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
                            
                            # График по сегментам
                            segment_counts = rec_df['segment'].value_counts()
                            fig2 = px.pie(
                                values=segment_counts.values,
                                names=segment_counts.index,
                                title="Распределение рекомендаций по сегментам"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        batch_top_k = st.slider("Рекомендаций на магазин:", 5, 15, 10)
                    with col2:
                        show_top_n = st.slider("Показать топ для отчета:", 3, 10, 5)
                    
                    if st.button("Сгенерировать рекомендации для всех магазинов"):
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
                                return df.to_csv(index=False).encode('utf-8')
                            
                            csv = convert_df(summary_df)
                            st.download_button(
                                label="📥 Скачать рекомендации (CSV)",
                                data=csv,
                                file_name='ensemble_recommendations.csv',
                                mime='text/csv'
                            )
                
                with tab3:
                    if st.session_state.recommender.processed_data is not None:
                        data = st.session_state.recommender.processed_data
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Распределение рейтингов
                            fig1 = px.histogram(
                                data, x='rating', bins=20,
                                title="Распределение рейтингов товаров"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Сегменты по рейтингу
                            segment_rating = data.groupby('Segment')['rating'].mean().sort_values(ascending=False)
                            fig2 = px.bar(
                                x=segment_rating.index, y=segment_rating.values,
                                title="Средний рейтинг по сегментам"
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Корреляционная матрица
                        numeric_cols = ['qty_sum', 'revenue_sum', 'price_mean', 'freq', 'rating']
                        corr_matrix = data[numeric_cols].corr()
                        
                        fig3 = px.imshow(
                            corr_matrix, 
                            title="Корреляции между метриками",
                            aspect="auto"
                        )
                        st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            st.error("Проверьте формат данных и попробуйте снова.")
    
    else:
        st.info("👆 Загрузите Excel файл для начала работы")
        
        # Пример структуры данных
        st.markdown("### 📋 Пример структуры данных:")
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
        
        st.markdown("### 🔧 Технические особенности:")
        st.markdown("""
        - **SVD**: Матричная факторизация для скрытых паттернов
        - **NMF**: Неотрицательная факторизация для интерпретируемости  
        - **Item-based CF**: Рекомендации на основе похожести товаров
        - **Content-based**: Учет характеристик товаров через Random Forest
        - **Ансамблирование**: Взвешенное объединение всех подходов
        """)

if __name__ == "__main__":
    create_dashboard()