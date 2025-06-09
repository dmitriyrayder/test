import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class OptimizedRecommenderSystem:
    def __init__(self):
        self.svd_model = None
        self.item_similarity = None
        self.user_item_matrix = None
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.processed_data = None
        self.weights = {'svd': 0.6, 'similarity': 0.4}  # Упрощенные веса
        
    def preprocess_data(self, df):
        """Упрощенная предобработка данных"""
        df = df.copy()
        
        # Базовая очистка
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        df = df[df['Price'] > 0]
        df = df[df['Qty'] > 0]
        
        # Создание рейтинга на основе количества и выручки
        df['Revenue'] = df['Price'] * df['Qty']
        
        # Энкодинг
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'])
        df['art_encoded'] = self.le_art.fit_transform(df['Art'])
        
        # Простая агрегация по магазин-товар
        agg_data = df.groupby(['magazin_encoded', 'art_encoded', 'Magazin', 'Art']).agg({
            'Qty': 'sum',
            'Revenue': 'sum',
            'Price': 'mean',
            'Segment': 'first',
            'Model': 'first'
        }).reset_index()
        
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
    
    def build_model(self, df, test_size=0.2):
        """Построение упрощенной модели"""
        # Предобработка
        df = self.preprocess_data(df)
        user_item_matrix = self.create_user_item_matrix(df)
        
        # Разделение данных для оценки
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Преобразование в плотную матрицу для SVD
        dense_matrix = user_item_matrix.toarray()
        
        # SVD (Matrix Factorization)
        n_components = min(30, min(dense_matrix.shape) - 1)  # Уменьшено количество компонент
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(dense_matrix)
        
        # Item-based Collaborative Filtering (только для топ товаров)
        # Берем только товары с достаточным количеством оценок для ускорения
        item_counts = np.array(user_item_matrix.sum(axis=0))[0]
        top_items_mask = item_counts >= np.percentile(item_counts[item_counts > 0], 50)
        
        if np.sum(top_items_mask) > 0:
            filtered_matrix = dense_matrix[:, top_items_mask]
            self.item_similarity = cosine_similarity(filtered_matrix.T)
            self.top_items_indices = np.where(top_items_mask)[0]
        else:
            self.item_similarity = cosine_similarity(dense_matrix.T)
            self.top_items_indices = np.arange(dense_matrix.shape[1])
        
        # Быстрая оценка качества
        sample_size = min(1000, len(test_data))
        test_sample = test_data.sample(n=sample_size, random_state=42)
        
        predictions = []
        actuals = []
        
        for _, row in test_sample.iterrows():
            pred = self.predict_single_rating(row['magazin_encoded'], row['art_encoded'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions)) ** 2))
        
        return {
            'rmse': rmse,
            'n_users': len(df['magazin_encoded'].unique()),
            'n_items': len(df['art_encoded'].unique()),
            'sparsity': 1 - user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        }
    
    def predict_single_rating(self, user_id, item_id):
        """Быстрое предсказание рейтинга"""
        predictions = []
        dense_matrix = self.user_item_matrix.toarray()
        
        # SVD предсказание
        if self.svd_model and user_id < dense_matrix.shape[0] and item_id < dense_matrix.shape[1]:
            user_factors = self.svd_model.transform(dense_matrix[user_id:user_id+1])
            item_factors = self.svd_model.components_[:, item_id]
            svd_pred = np.dot(user_factors[0], item_factors)
            predictions.append(('svd', svd_pred))
        
        # Item similarity предсказание (только для отфильтрованных товаров)
        if (hasattr(self, 'top_items_indices') and 
            user_id < dense_matrix.shape[0] and 
            item_id in self.top_items_indices):
            
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
        
        # Ансамблевое предсказание
        if predictions:
            weighted_sum = sum(pred * self.weights[method] for method, pred in predictions)
            total_weight = sum(self.weights[method] for method, _ in predictions)
            return weighted_sum / total_weight if total_weight > 0 else 2.5
        
        return 2.5
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Быстрое получение рекомендаций"""
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
            pred_rating = self.predict_single_rating(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # Сортировка и выбор топ-K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = predictions[:top_k]
        
        # Формирование результата
        recommendations = []
        for rank, (item_id, score) in enumerate(top_items, 1):
            try:
                item_name = self.le_art.inverse_transform([item_id])[0]
                
                # Поиск информации о товаре
                item_info = self.processed_data[self.processed_data['art_encoded'] == item_id]
                
                if len(item_info) > 0:
                    info = item_info.iloc[0]
                    recommendations.append({
                        'rank': rank,
                        'item': item_name,
                        'score': round(score, 3),
                        'segment': info['Segment'],
                        'model': info['Model'],
                        'avg_price': round(info['Price'], 2),
                        'total_qty': int(info['Qty'])
                    })
                else:
                    recommendations.append({
                        'rank': rank,
                        'item': item_name,
                        'score': round(score, 3),
                        'segment': 'Unknown',
                        'model': 'Unknown',
                        'avg_price': 0,
                        'total_qty': 0
                    })
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
    
    st.title("🛍️ Оптимизированная рекомендательная система")
    st.markdown("*SVD + Item-based коллаборативная фильтрация*")
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
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Проверка колонок
            required_cols = ['Magazin', 'Art', 'Segment', 'Model', 'Price', 'Qty']
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
                with st.spinner("Обучение модели..."):
                    metrics = st.session_state.recommender.build_model(df)
                
                st.success("Модель обучена!")
                
                # Метрики модели
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.3f}")
                with col2:
                    st.metric("Разреженность", f"{metrics['sparsity']:.1%}")
                with col3:
                    st.metric("Пользователи x Товары", f"{metrics['n_users']} x {metrics['n_items']}")
            
            # Рекомендации
            if st.session_state.recommender.user_item_matrix is not None:
                st.markdown("---")
                st.header("📊 Рекомендации")
                
                tab1, tab2 = st.tabs(["Для одного магазина", "Для всех магазинов"])
                
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
                            st.dataframe(rec_df, use_container_width=True)
                            
                            # Статистика
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_score = rec_df['score'].mean()
                                st.metric("Средний прогноз", f"{avg_score:.3f}")
                            with col2:
                                top_segment = rec_df['segment'].mode().iloc[0] if len(rec_df) > 0 else "N/A"
                                st.metric("Топ сегмент", top_segment)
                            with col3:
                                avg_price = rec_df['avg_price'].mean()
                                st.metric("Средняя цена", f"{avg_price:.2f}")
                        else:
                            st.info("Нет рекомендаций для данного магазина")
                
                with tab2:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        batch_top_k = st.slider("Рекомендаций на магазин:", 5, 15, 10)
                    with col2:
                        show_top_n = st.slider("Показать топ для отчета:", 3, 10, 5)
                    
                    if st.button("Сгенерировать рекомендации для всех"):
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
                                        'Прогноз': rec['score'],
                                        'Сегмент': rec['segment'],
                                        'Модель': rec['model'],
                                        'Цена': rec['avg_price'],
                                        'Количество': rec['total_qty']
                                    })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Статистика
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Всего рекомендаций", len(summary_data))
                            with col2:
                                avg_score = summary_df['Прогноз'].mean()
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
                                file_name='recommendations.csv',
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

if __name__ == "__main__":
    create_dashboard()
