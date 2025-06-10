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
        self.content_features = None
        self.weights = {'svd': 0.4, 'nmf': 0.3, 'similarity': 0.2, 'content': 0.1}
        
    def preprocess_data(self, df):
        """Предобработка данных"""
        df = df.copy()
        
        # Проверка наличия обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        # Преобразование даты с обработкой ошибок
        try:
            df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        except Exception as e:
            raise ValueError(f"Ошибка преобразования даты: {e}")
        
        # Удаление строк с некорректными данными
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty', 'Datasales'])
        
        if len(df) == 0:
            raise ValueError("После очистки данных не осталось строк")
        
        # Создание временных признаков
        df['Month'] = df['Datasales'].dt.month
        df['Quarter'] = df['Datasales'].dt.quarter
        df['Weekday'] = df['Datasales'].dt.dayofweek
        df['DayOfMonth'] = df['Datasales'].dt.day
        
        # Бизнес-метрики
        df['Revenue'] = df['Price'] * df['Qty']
        
        # Безопасное создание ценовых категорий
        try:
            df['PriceCategory'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        except Exception:
            df['PriceCategory'] = 'Medium'  # Значение по умолчанию
        
        # Энкодинг с проверкой
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'].astype(str))
        df['art_encoded'] = self.le_art.fit_transform(df['Art'].astype(str))
        
        # Заполнение отсутствующих значений для категориальных переменных
        categorical_cols = ['Segment', 'Model', 'Describe']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = 'Unknown'
        
        # Агрегация по магазин-товар
        try:
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
        except Exception as e:
            raise ValueError(f"Ошибка агрегации данных: {e}")
        
        # Создание рейтинга с проверкой на нулевые значения
        agg_data['qty_sum'] = np.maximum(agg_data['qty_sum'], 1e-8)
        agg_data['revenue_sum'] = np.maximum(agg_data['revenue_sum'], 1e-8)
        agg_data['freq'] = np.maximum(agg_data['freq'], 1)
        
        agg_data['rating'] = (
            np.log1p(agg_data['qty_sum']) * 0.4 +
            np.log1p(agg_data['revenue_sum']) * 0.4 +
            np.log1p(agg_data['freq']) * 0.2
        )
        
        # Нормализация рейтинга с проверкой на постоянные значения
        rating_min = agg_data['rating'].min()
        rating_max = agg_data['rating'].max()
        
        if rating_max - rating_min > 1e-8:
            agg_data['rating'] = (agg_data['rating'] - rating_min) / (rating_max - rating_min) * 4 + 1
        else:
            agg_data['rating'] = 2.5  # Средний рейтинг если все значения одинаковые
        
        self.processed_data = agg_data
        return agg_data
    
    def create_user_item_matrix(self, df):
        """Создание матрицы пользователь-товар"""
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        if n_users == 0 or n_items == 0:
            raise ValueError("Нет данных для создания матрицы пользователь-товар")
        
        # Создание разреженной матрицы
        try:
            user_item_matrix = csr_matrix((df['rating'], 
                                         (df['magazin_encoded'], df['art_encoded'])), 
                                        shape=(n_users, n_items))
            
            self.user_item_matrix = user_item_matrix.toarray()
        except Exception as e:
            raise ValueError(f"Ошибка создания матрицы: {e}")
        
        return self.user_item_matrix
    
    def prepare_content_features(self, df):
        """Подготовка контентных признаков"""
        try:
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
            
            # Проверка на наличие числовых данных
            for col in numeric_cols:
                if features[col].std() > 1e-8:  # Проверка на вариативность
                    features[[col]] = self.scaler.fit_transform(features[[col]])
                else:
                    features[col] = 0  # Нормализация к нулю если нет вариативности
            
            self.content_features = features
            return features
        
        except Exception as e:
            st.warning(f"Ошибка подготовки контентных признаков: {e}")
            return pd.DataFrame()
    
    def build_ensemble_model(self, df, test_size=0.2):
        """Построение ансамбля моделей"""
        try:
            # Предобработка
            df = self.preprocess_data(df)
            user_item_matrix = self.create_user_item_matrix(df)
            content_features = self.prepare_content_features(df)
            
            if len(df) < 10:
                raise ValueError("Недостаточно данных для обучения модели (минимум 10 записей)")
            
            # Разделение данных
            train_data, test_data = train_test_split(df, test_size=min(test_size, 0.5), random_state=42)
            
            # 1. SVD (Matrix Factorization)
            n_components_svd = min(50, min(user_item_matrix.shape) - 1)
            if n_components_svd > 0:
                self.svd_model = TruncatedSVD(n_components=n_components_svd, random_state=42)
                svd_matrix = self.svd_model.fit_transform(user_item_matrix)
            
            # 2. NMF (Non-negative Matrix Factorization)
            n_components_nmf = min(30, min(user_item_matrix.shape) - 1)
            if n_components_nmf > 0:
                try:
                    self.nmf_model = NMF(n_components=n_components_nmf, random_state=42, max_iter=500)
                    nmf_matrix = self.nmf_model.fit_transform(np.maximum(user_item_matrix, 0))
                except Exception as e:
                    st.warning(f"Ошибка NMF: {e}")
                    self.nmf_model = None
            
            # 3. Item-based Collaborative Filtering
            if user_item_matrix.shape[1] > 1:
                self.item_similarity = cosine_similarity(user_item_matrix.T)
            
            # 4. Content-based Random Forest
            if len(content_features) > 0 and len(self.feature_columns) > 0:
                try:
                    # Подготовка данных для RF
                    rf_data = df.merge(content_features, on='art_encoded', how='left')
                    X_rf = rf_data[self.feature_columns].fillna(0)
                    y_rf = rf_data['rating']
                    
                    if len(X_rf) > 5:  # Минимум данных для RF
                        self.rf_model = RandomForestRegressor(
                            n_estimators=min(100, len(X_rf) * 2), 
                            random_state=42, 
                            max_depth=min(10, len(self.feature_columns))
                        )
                        self.rf_model.fit(X_rf, y_rf)
                except Exception as e:
                    st.warning(f"Ошибка Random Forest: {e}")
                    self.rf_model = None
            
            # Вычисление метрик
            try:
                train_predictions = self.predict_ratings_for_evaluation(train_data)
                test_predictions = self.predict_ratings_for_evaluation(test_data)
                
                train_rmse = np.sqrt(np.mean((train_data['rating'] - train_predictions) ** 2))
                test_rmse = np.sqrt(np.mean((test_data['rating'] - test_predictions) ** 2))
            except Exception as e:
                st.warning(f"Ошибка вычисления метрик: {e}")
                train_rmse = test_rmse = 0.0
            
            return {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'n_users': len(df['magazin_encoded'].unique()),
                'n_items': len(df['art_encoded'].unique()),
                'sparsity': 1 - np.count_nonzero(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
            }
            
        except Exception as e:
            raise Exception(f"Ошибка построения модели: {e}")
    
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
        
        # Проверка границ
        if (self.user_item_matrix is None or 
            user_id >= self.user_item_matrix.shape[0] or 
            item_id >= self.user_item_matrix.shape[1]):
            return 2.5
        
        # SVD предсказание
        if self.svd_model is not None:
            try:
                user_factors = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_factors = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('svd', svd_pred))
            except Exception:
                pass
        
        # NMF предсказание
        if self.nmf_model is not None:
            try:
                user_matrix = np.maximum(self.user_item_matrix[user_id:user_id+1], 0)
                user_factors = self.nmf_model.transform(user_matrix)
                item_factors = self.nmf_model.components_[:, item_id]
                nmf_pred = np.dot(user_factors, item_factors.reshape(-1, 1))[0, 0]
                predictions.append(('nmf', nmf_pred))
            except Exception:
                pass
        
        # Item similarity предсказание
        if self.item_similarity is not None:
            try:
                user_ratings = self.user_item_matrix[user_id]
                similar_items = self.item_similarity[item_id]
                
                # Взвешенное среднее по похожим товарам
                numerator = np.sum(similar_items * user_ratings)
                denominator = np.sum(np.abs(similar_items))
                
                if denominator > 1e-8:
                    similarity_pred = numerator / denominator
                    predictions.append(('similarity', similarity_pred))
            except Exception:
                pass
        
        # Content-based предсказание
        if self.rf_model is not None and self.content_features is not None:
            try:
                item_features = self.content_features[
                    self.content_features['art_encoded'] == item_id
                ]
                if len(item_features) > 0:
                    X_content = item_features[self.feature_columns].fillna(0)
                    content_pred = self.rf_model.predict(X_content)[0]
                    predictions.append(('content', content_pred))
            except Exception:
                pass
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights.get(method, 0.25) for method, pred in predictions)
            total_weight = sum(self.weights.get(method, 0.25) for method, _ in predictions)
            final_pred = weighted_sum / total_weight if total_weight > 0 else 2.5
            
            # Ограничение предсказания разумными пределами
            return np.clip(final_pred, 1.0, 5.0)
        
        return 2.5  # Средний рейтинг по умолчанию
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Получение рекомендаций для магазина"""
        if self.user_item_matrix is None or self.processed_data is None:
            return None
        
        try:
            user_id = self.le_magazin.transform([magazin_name])[0]
        except Exception:
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
                try:
                    pred_rating = self.predict_single_rating(user_id, item_id)
                    predictions.append((item_id, pred_rating))
                except Exception:
                    continue
        
        if not predictions:
            return None
        
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
            except Exception:
                continue
        
        return recommendations if recommendations else None
    
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
            except Exception:
                continue
        
        return all_recommendations if all_recommendations else None

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
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Проверка колонок
            required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют обязательные колонки: {missing_cols}")
                st.info("Обязательные колонки: Magazin, Datasales, Art, Price, Qty")
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
                segment_count = df['Segment'].nunique() if 'Segment' in df.columns else 0
                st.metric("Сегментов", segment_count)
            
            # Построение модели
            if st.sidebar.button("🚀 Построить модель", type="primary"):
                try:
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
                        overfitting = max(0, metrics['test_rmse'] - metrics['train_rmse'])
                        st.metric("Переобучение", f"{overfitting:.3f}")
                        
                except Exception as e:
                    st.error(f"Ошибка построения модели: {str(e)}")
                    st.info("Проверьте качество данных и попробуйте снова")
            
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
                            if 'segment' in rec_df.columns:
                                segment_counts = rec_df['segment'].value_counts()
                                if len(segment_counts) > 0:
                                    fig2 = px.pie(
                                        values=segment_counts.values,
                                        names=segment_counts.index,
                                        title="Распределение рекомендаций по сегментам"
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.warning("Не удалось сгенерировать рекомендации для выбранного магазина")
                
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
                                    return df.to_csv(index=False).encode('utf-8')
                                
                                csv = convert_df(summary_df)
                                st.download_button(
                                    label="📥 Скачать рекомендации (CSV)",
                                    data=csv,
                                    file_name='ensemble_recommendations.csv',
                                    mime='text/csv'
                                )
                            else:
                                st.warning("Не удалось создать сводную таблицу рекомендаций")
                        else:
                            st.warning("Не удалось сгенерировать рекомендации")
                
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
                            if 'Segment' in data.columns and data['Segment'].nunique() > 1:
                                segment_rating = data.groupby('Segment')['rating'].mean().sort_values(ascending=False)
                                fig2 = px.bar(
                                    x=segment_rating.index, y=segment_rating.values,
                                    title="Средний рейтинг по сегментам"
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("Недостаточно данных для анализа по сегментам")
                        
                        # Корреляционная матрица
                        numeric_cols = ['qty_sum', 'revenue_sum', 'price_mean', 'freq', 'rating']
                        available_cols = [col for col in numeric_cols if col in data.columns]
                        
                        if len(available_cols) > 1:
                            corr_matrix = data[available_cols].corr()
                            
                            fig3 = px.imshow(
                                corr_matrix, 
                                title="Корреляции между метриками",
                                aspect="auto",
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                        else:
                            st.info("Недостаточно числовых колонок для корреляционного анализа")
                    else:
                        st.info("Данные не обработаны. Постройте модель сначала.")
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            st.error("Проверьте формат данных и попробуйте снова.")
            
            # Показать детали ошибки в режиме отладки
            if st.checkbox("Показать детали ошибки"):
                st.exception(e)
    
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
        
        st.markdown("### ⚠️ Требования к данным:")
        st.markdown("""
        - **Обязательные колонки**: Magazin, Datasales, Art, Price, Qty
        - **Дополнительные колонки**: Describe, Model, Segment (будут заполнены 'Unknown' если отсутствуют)
        - **Формат даты**: Любой стандартный формат даты
        - **Числовые данные**: Price и Qty должны быть числовыми
        - **Минимум данных**: Рекомендуется не менее 100 записей для качественной работы
        """)

if __name__ == "__main__":
    create_dashboard()
