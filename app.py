import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class RecommenderSystem:
    def __init__(self):
        self.svd_model = None
        self.item_similarity = None
        self.rf_model = None
        self.user_item_matrix = None
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.processed_data = None
        self.weights = {'svd': 0.4, 'similarity': 0.4, 'content': 0.2}
        
    def preprocess_data(self, df, selected_segments=None):
        """Предобработка данных с фильтрацией по сегментам"""
        df = df.copy()
        
        # Проверка обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")
        
        # Фильтрация по сегментам
        if selected_segments and 'Segment' in df.columns:
            df = df[df['Segment'].isin(selected_segments)]
            if len(df) == 0:
                raise ValueError("Нет данных после фильтрации сегментов")
        
        # Очистка данных
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        df = df.dropna(subset=['Datasales'])
        
        # Создание признаков
        df['Revenue'] = df['Price'] * df['Qty']
        df['Month'] = df['Datasales'].dt.month
        
        # Заполнение пустых значений
        for col in ['Segment', 'Model', 'Describe']:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown')
        
        # Энкодинг
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'].astype(str))
        df['art_encoded'] = self.le_art.fit_transform(df['Art'].astype(str))
        
        # Агрегация по магазин-товар (на основе штучных продаж)
        agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg({
            'Qty': ['sum', 'mean', 'count'],
            'Revenue': 'sum',
            'Price': 'mean',
            'Segment': 'first',
            'Model': 'first'
        }).reset_index()
        
        # Упрощение колонок
        agg_data.columns = ['magazin_encoded', 'art_encoded', 'Magazin', 'Art', 
                           'qty_sum', 'qty_mean', 'freq', 'revenue_sum', 'price_mean',
                           'Segment', 'Model']
        
        # Создание рейтинга на основе штучных продаж
        agg_data['qty_sum'] = np.maximum(agg_data['qty_sum'], 1)
        agg_data['freq'] = np.maximum(agg_data['freq'], 1)
        
        # Рейтинг = нормализованная функция от количества и частоты
        agg_data['rating'] = (
            np.log1p(agg_data['qty_sum']) * 0.7 +  # Больший вес на количество
            np.log1p(agg_data['freq']) * 0.3
        )
        
        # Нормализация рейтинга в диапазон 1-5
        rating_min, rating_max = agg_data['rating'].min(), agg_data['rating'].max()
        if rating_max > rating_min:
            agg_data['rating'] = (agg_data['rating'] - rating_min) / (rating_max - rating_min) * 4 + 1
        else:
            agg_data['rating'] = 2.5
        
        self.processed_data = agg_data
        return agg_data
    
    def create_user_item_matrix(self, df):
        """Создание матрицы пользователь-товар"""
        n_users = df['magazin_encoded'].nunique()
        n_items = df['art_encoded'].nunique()
        
        user_item_matrix = csr_matrix(
            (df['rating'], (df['magazin_encoded'], df['art_encoded'])), 
            shape=(n_users, n_items)
        )
        
        self.user_item_matrix = user_item_matrix.toarray()
        return self.user_item_matrix
    
    def build_model(self, df, selected_segments=None):
        """Построение модели"""
        # Предобработка
        df = self.preprocess_data(df, selected_segments)
        if len(df) < 10:
            raise ValueError("Недостаточно данных (минимум 10 записей)")
        
        user_item_matrix = self.create_user_item_matrix(df)
        
        # Разделение данных
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        
        # 1. SVD
        n_components = min(30, min(user_item_matrix.shape) - 1)
        if n_components > 0:
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_model.fit(user_item_matrix)
        
        # 2. Item Similarity
        if user_item_matrix.shape[1] > 1:
            self.item_similarity = cosine_similarity(user_item_matrix.T)
        
        # 3. Content-based Random Forest
        try:
            # Создание контентных признаков
            item_features = df.groupby('art_encoded').agg({
                'price_mean': 'first',
                'qty_mean': 'first',
                'Segment': 'first'
            }).reset_index()
            
            # One-hot encoding для сегментов
            segment_dummies = pd.get_dummies(item_features['Segment'], prefix='segment')
            
            # Объединение признаков
            features_df = pd.concat([
                item_features[['art_encoded', 'price_mean', 'qty_mean']],
                segment_dummies
            ], axis=1)
            
            # Подготовка данных для RF
            rf_data = df.merge(features_df, on='art_encoded', how='left')
            feature_cols = [col for col in features_df.columns if col != 'art_encoded']
            
            X = rf_data[feature_cols].fillna(0)
            y = rf_data['rating']
            
            self.rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            self.rf_model.fit(X, y)
            self.feature_columns = feature_cols
            self.content_features = features_df
            
        except Exception as e:
            st.warning(f"Content-based модель не построена: {e}")
            self.rf_model = None
        
        # Вычисление метрик
        train_pred = self.predict_batch(train_data)
        test_pred = self.predict_batch(test_data)
        
        train_rmse = np.sqrt(np.mean((train_data['rating'] - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((test_data['rating'] - test_pred) ** 2))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'n_users': len(df['magazin_encoded'].unique()),
            'n_items': len(df['art_encoded'].unique()),
            'sparsity': 1 - np.count_nonzero(user_item_matrix) / user_item_matrix.size
        }
    
    def predict_batch(self, data):
        """Предсказание для батча данных"""
        predictions = []
        for _, row in data.iterrows():
            pred = self.predict_rating(row['magazin_encoded'], row['art_encoded'])
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_rating(self, user_id, item_id):
        """Предсказание рейтинга"""
        if (user_id >= self.user_item_matrix.shape[0] or 
            item_id >= self.user_item_matrix.shape[1]):
            return 2.5
        
        predictions = []
        
        # SVD предсказание
        if self.svd_model is not None:
            try:
                user_vec = self.svd_model.transform(self.user_item_matrix[user_id:user_id+1])
                item_vec = self.svd_model.components_[:, item_id]
                svd_pred = np.dot(user_vec, item_vec.reshape(-1, 1))[0, 0]
                predictions.append(('svd', svd_pred))
            except:
                pass
        
        # Similarity предсказание
        if self.item_similarity is not None:
            try:
                user_ratings = self.user_item_matrix[user_id]
                similar_items = self.item_similarity[item_id]
                
                numerator = np.sum(similar_items * user_ratings)
                denominator = np.sum(np.abs(similar_items))
                
                if denominator > 1e-8:
                    sim_pred = numerator / denominator
                    predictions.append(('similarity', sim_pred))
            except:
                pass
        
        # Content предсказание
        if self.rf_model is not None:
            try:
                item_data = self.content_features[
                    self.content_features['art_encoded'] == item_id
                ]
                if len(item_data) > 0:
                    X = item_data[self.feature_columns].fillna(0)
                    content_pred = self.rf_model.predict(X)[0]
                    predictions.append(('content', content_pred))
            except:
                pass
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights.get(method, 0.33) for method, pred in predictions)
            total_weight = sum(self.weights.get(method, 0.33) for method, _ in predictions)
            return np.clip(weighted_sum / total_weight, 1.0, 5.0)
        
        return 2.5
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Получение рекомендаций"""
        try:
            user_id = self.le_magazin.transform([magazin_name])[0]
        except:
            return None
        
        if user_id >= self.user_item_matrix.shape[0]:
            return None
        
        user_ratings = self.user_item_matrix[user_id]
        predictions = []
        
        for item_id in range(self.user_item_matrix.shape[1]):
            if user_ratings[item_id] == 0:  # Только неоцененные товары
                pred = self.predict_rating(user_id, item_id)
                predictions.append((item_id, pred))
        
        if not predictions:
            return None
        
        # Сортировка и выбор топ-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = predictions[:top_k]
        
        recommendations = []
        for rank, (item_id, score) in enumerate(top_items, 1):
            try:
                item_name = self.le_art.inverse_transform([item_id])[0]
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

def create_dashboard():
    st.set_page_config(page_title="Рекомендательная система", layout="wide")
    
    st.title("🛍️ Рекомендательная система")
    st.markdown("*Ансамбль: SVD + Коллаборативная фильтрация + Content-based*")
    st.markdown("---")
    
    # Инициализация
    if 'recommender' not in st.session_state:
        st.session_state.recommender = RecommenderSystem()
    
    # Загрузка файла
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Проверка колонок
            required_cols = ['Magazin', 'Datasales', 'Art', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
                return
            
            # Информация о данных
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Записей", len(df))
            with col2:
                st.metric("Магазинов", df['Magazin'].nunique())
            with col3:
                st.metric("Товаров", df['Art'].nunique())
            with col4:
                segments = df['Segment'].nunique() if 'Segment' in df.columns else 0
                st.metric("Сегментов", segments)
            
            # Фильтр сегментов
            segment_filter = None
            if 'Segment' in df.columns and df['Segment'].nunique() > 1:
                st.sidebar.header("🎯 Фильтр сегментов")
                all_segments = df['Segment'].dropna().unique().tolist()
                selected_segments = st.sidebar.multiselect(
                    "Выберите сегменты:",
                    options=all_segments,
                    default=all_segments,
                    help="Выберите сегменты для обучения модели"
                )
                
                if selected_segments != all_segments:
                    segment_filter = selected_segments
                    filtered_count = len(df[df['Segment'].isin(selected_segments)])
                    st.sidebar.info(f"Записей после фильтрации: {filtered_count}")
            
            # Построение модели
            if st.sidebar.button("🚀 Построить модель", type="primary"):
                try:
                    with st.spinner("Обучение модели..."):
                        metrics = st.session_state.recommender.build_model(df, segment_filter)
                    
                    st.success("Модель обучена!")
                    
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
                    st.error(f"Ошибка: {str(e)}")
            
            # Рекомендации
            if st.session_state.recommender.user_item_matrix is not None:
                st.markdown("---")
                st.header("📊 Рекомендации")
                
                tab1, tab2, tab3 = st.tabs(["Одиночные", "Массовые", "Аналитика"])
                
                with tab1:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        selected_shop = st.selectbox(
                            "Магазин:",
                            options=st.session_state.recommender.le_magazin.classes_
                        )
                    with col2:
                        top_k = st.slider("Количество:", 5, 20, 10)
                    
                    if st.button("Получить рекомендации"):
                        recs = st.session_state.recommender.get_recommendations(selected_shop, top_k)
                        
                        if recs:
                            rec_df = pd.DataFrame(recs)
                            
                            # Форматирование
                            display_df = rec_df.copy()
                            display_df['score'] = display_df['score'].round(3)
                            display_df['avg_price'] = display_df['avg_price'].round(2)
                            display_df['expected_qty'] = display_df['expected_qty'].round(1)
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # График
                            fig = px.bar(
                                rec_df.head(10), x='item', y='score',
                                title=f"Топ-10 для {selected_shop}",
                                color='score',
                                color_continuous_scale='viridis'
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Нет рекомендаций")
                
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        batch_k = st.slider("Рекомендаций на магазин:", 5, 15, 8)
                    with col2:
                        show_top = st.slider("Показать в отчете:", 3, 10, 5)
                    
                    if st.button("Генерировать для всех"):
                        summary_data = []
                        
                        with st.spinner("Генерация..."):
                            for shop in st.session_state.recommender.le_magazin.classes_:
                                recs = st.session_state.recommender.get_recommendations(shop, batch_k)
                                if recs:
                                    for rec in recs[:show_top]:
                                        summary_data.append({
                                            'Магазин': shop,
                                            'Ранг': rec['rank'],
                                            'Товар': rec['item'],
                                            'Прогноз': f"{rec['score']:.3f}",
                                            'Сегмент': rec['segment'],
                                            'Цена': f"{rec['avg_price']:.2f}"
                                        })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Всего", len(summary_data))
                            with col2:
                                avg_score = np.mean([float(x) for x in summary_df['Прогноз']])
                                st.metric("Средний прогноз", f"{avg_score:.3f}")
                            with col3:
                                st.metric("Уникальных товаров", summary_df['Товар'].nunique())
                            
                            # Скачивание
                            csv = summary_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Скачать CSV",
                                data=csv,
                                file_name='recommendations.csv',
                                mime='text/csv'
                            )
                
                with tab3:
                    if st.session_state.recommender.processed_data is not None:
                        data = st.session_state.recommender.processed_data
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Исправленный histogram
                            fig1 = px.histogram(
                                data, x='rating', nbins=20,
                                title="Распределение рейтингов"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            if 'Segment' in data.columns and data['Segment'].nunique() > 1:
                                segment_rating = data.groupby('Segment')['rating'].mean().sort_values(ascending=False)
                                fig2 = px.bar(
                                    x=segment_rating.index, y=segment_rating.values,
                                    title="Рейтинг по сегментам"
                                )
                                st.plotly_chart(fig2, use_container_width=True)
                        
                        # Топ товары по продажам
                        top_items = data.nlargest(10, 'qty_sum')[['Art', 'qty_sum', 'Segment']]
                        fig3 = px.bar(
                            top_items, x='Art', y='qty_sum', color='Segment',
                            title="Топ-10 товаров по продажам"
                        )
                        fig3.update_xaxes(tickangle=45)
                        st.plotly_chart(fig3, use_container_width=True)
        
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
    
    else:
        st.info("👆 Загрузите Excel файл")
        
        # Пример данных
        st.markdown("### 📋 Пример структуры:")
        example = pd.DataFrame({
            'Magazin': ['Shop_A', 'Shop_B', 'Shop_A'],
            'Datasales': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Art': ['Item_001', 'Item_002', 'Item_001'],
            'Segment': ['Electronics', 'Clothing', 'Electronics'],
            'Price': [100, 50, 100],
            'Qty': [2, 1, 3]
        })
        st.dataframe(example, use_container_width=True)

if __name__ == "__main__":
    create_dashboard()
