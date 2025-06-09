import streamlit as st
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from scipy.sparse import csr_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn. preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io
import warnings
warnings.filterwarnings('ignore')

class ShopRecommenderSystem:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.item_features = None
        self.user_features = None
        self.item_id_map = {}
        self.user_id_map = {}
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.processed_data = None
        
    def preprocess_data(self, df):
        """Предобработка данных"""
        # Очистка и подготовка данных
        df = df.copy()
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty'])
        
        # Создание признаков
        df['Revenue'] = df['Price'] * df['Qty']
        df['Month'] = df['Datasales'].dt.month
        df['Quarter'] = df['Datasales'].dt.quarter
        df['Weekday'] = df['Datasales'].dt.dayofweek
        
        # Энкодинг категориальных переменных
        df['magazin_encoded'] = self.le_magazin.fit_transform(df['Magazin'])
        df['art_encoded'] = self.le_art.fit_transform(df['Art'])
        
        # Агрегация данных по магазину-товару
        agg_data = df.groupby(['magazin_encoded', 'art_encoded']).agg({
            'Qty': 'sum',
            'Revenue': 'sum',
            'Price': 'mean',
            'Segment': 'first',
            'Model': 'first',
            'Describe': 'first'
        }).reset_index()
        
        # Создание рейтинга (можно настроить формулу)
        agg_data['rating'] = np.log1p(agg_data['Qty']) * np.log1p(agg_data['Revenue'] / agg_data['Price'])
        
        self.processed_data = agg_data
        return agg_data
    
    def prepare_features(self, df):
        """Подготовка признаков для LightFM"""
        # Признаки товаров
        item_features = df[['art_encoded', 'Segment', 'Model']].drop_duplicates()
        item_features = item_features.fillna('Unknown')
        
        # Признаки пользователей (магазинов) - можно расширить
        user_features = df[['magazin_encoded']].drop_duplicates()
        
        return item_features, user_features
    
    def build_model(self, df, test_size=0.2):
        """Построение модели LightFM"""
        # Подготовка данных
        df = self.preprocess_data(df)
        item_features, user_features = self.prepare_features(df)
        
        # Создание датасета LightFM
        self.dataset = Dataset()
        
        # Подготовка признаков
        item_feature_list = []
        for _, row in item_features.iterrows():
            features = [f"segment_{row['Segment']}", f"model_{row['Model']}"]
            item_feature_list.append((row['art_encoded'], features))
        
        # Инициализация датасета
        self.dataset.fit(
            users=df['magazin_encoded'].unique(),
            items=df['art_encoded'].unique(),
            item_features=[f"segment_{s}" for s in df['Segment'].unique()] + 
                         [f"model_{m}" for m in df['Model'].unique()]
        )
        
        # Создание матриц взаимодействий
        interactions, weights = self.dataset.build_interactions(
            [(row['magazin_encoded'], row['art_encoded'], row['rating']) 
             for _, row in df.iterrows()]
        )
        
        # Создание матрицы признаков товаров
        item_features_matrix = self.dataset.build_item_features(item_feature_list)
        
        # Разделение на train/test
        train_interactions, test_interactions = train_test_split(
            list(zip(df['magazin_encoded'], df['art_encoded'], df['rating'])),
            test_size=test_size, random_state=42
        )
        
        train_matrix, _ = self.dataset.build_interactions(train_interactions)
        test_matrix, _ = self.dataset.build_interactions(test_interactions)
        
        # Обучение модели
        self.model = LightFM(loss='warp', random_state=42)
        self.model.fit(train_matrix, item_features=item_features_matrix, epochs=30, num_threads=2)
        
        # Оценка качества
        train_precision = precision_at_k(self.model, train_matrix, k=10).mean()
        test_precision = precision_at_k(self.model, test_matrix, k=10).mean()
        
        self.item_features = item_features_matrix
        
        return {
            'train_precision': train_precision,
            'test_precision': test_precision,
            'n_users': len(df['magazin_encoded'].unique()),
            'n_items': len(df['art_encoded'].unique())
        }
    
    def get_recommendations(self, magazin_name, top_k=10):
        """Получение рекомендаций для магазина"""
        if self.model is None:
            return None
        
        try:
            magazin_encoded = self.le_magazin.transform([magazin_name])[0]
        except:
            return None
        
        # Получение всех товаров
        n_items = len(self.le_art.classes_)
        item_ids = np.arange(n_items)
        
        # Предсказание скоров
        scores = self.model.predict(magazin_encoded, item_ids, item_features=self.item_features)
        
        # Топ рекомендации
        top_items = np.argsort(-scores)[:top_k]
        top_scores = scores[top_items]
        
        # Получение названий товаров
        item_names = self.le_art.inverse_transform(top_items)
        
        # Добавление информации о товарах
        recommendations = []
        for i, (item_name, score) in enumerate(zip(item_names, top_scores)):
            item_info = self.processed_data[
                self.processed_data['art_encoded'] == top_items[i]
            ].iloc[0] if len(self.processed_data[
                self.processed_data['art_encoded'] == top_items[i]
            ]) > 0 else None
            
            rec = {
                'rank': i + 1,
                'item': item_name,
                'score': score,
                'segment': item_info['Segment'] if item_info is not None else 'Unknown',
                'model': item_info['Model'] if item_info is not None else 'Unknown',
                'avg_price': item_info['Price'] if item_info is not None else 0
            }
            recommendations.append(rec)
        
        return recommendations
    
    def get_all_recommendations(self, top_k=10):
        """Получение рекомендаций для всех магазинов"""
        if self.model is None:
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
    st.markdown("---")
    
    # Инициализация системы
    if 'recommender' not in st.session_state:
        st.session_state.recommender = ShopRecommenderSystem()
    
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
                with st.spinner("Обучение модели..."):
                    metrics = st.session_state.recommender.build_model(df)
                
                st.success("Модель обучена!")
                
                # Метрики модели
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precision@10 (train)", f"{metrics['train_precision']:.3f}")
                with col2:
                    st.metric("Precision@10 (test)", f"{metrics['test_precision']:.3f}")
            
            # Рекомендации
            if st.session_state.recommender.model is not None:
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
                            st.dataframe(rec_df, use_container_width=True)
                            
                            # График
                            fig = px.bar(
                                rec_df, x='item', y='score',
                                title=f"Рекомендации для {selected_shop}",
                                labels={'item': 'Товар', 'score': 'Скор'}
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if st.button("Сгенерировать рекомендации для всех магазинов"):
                        with st.spinner("Генерация рекомендаций..."):
                            all_recs = st.session_state.recommender.get_all_recommendations(10)
                        
                        if all_recs:
                            # Создание сводной таблицы
                            summary_data = []
                            for shop, recs in all_recs.items():
                                for rec in recs[:5]:  # Топ-5 для каждого магазина
                                    summary_data.append({
                                        'Магазин': shop,
                                        'Ранг': rec['rank'],
                                        'Товар': rec['item'],
                                        'Скор': f"{rec['score']:.3f}",
                                        'Сегмент': rec['segment'],
                                        'Модель': rec['model']
                                    })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Скачивание результатов
                            @st.cache_data
                            def convert_df(df):
                                return df.to_csv(index=False).encode('utf-8')
                            
                            csv = convert_df(summary_df)
                            st.download_button(
                                label="📥 Скачать CSV",
                                data=csv,
                                file_name='recommendations.csv',
                                mime='text/csv'
                            )
                
                with tab3:
                    if st.session_state.recommender.processed_data is not None:
                        data = st.session_state.recommender.processed_data
                        
                        # Распределение по сегментам
                        fig1 = px.pie(
                            data, names='Segment', 
                            title="Распределение товаров по сегментам"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Топ товары по рейтингу
                        top_items = data.nlargest(20, 'rating')
                        item_names = st.session_state.recommender.le_art.inverse_transform(top_items['art_encoded'])
                        top_items = top_items.copy()
                        top_items['item_name'] = item_names
                        
                        fig2 = px.bar(
                            top_items, x='item_name', y='rating',
                            title="Топ-20 товаров по рейтингу"
                        )
                        fig2.update_xaxes(tickangle=45)
                        st.plotly_chart(fig2, use_container_width=True)
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
    
    else:
        st.info("👆 Загрузите Excel файл для начала работы")
        
        # Пример структуры данных
        st.markdown("### 📋 Пример структуры данных:")
        example_data = {
            'Magazin': ['Shop_A', 'Shop_B', 'Shop_A'],
            'Datasales': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Art': ['Item_001', 'Item_002', 'Item_003'],
            'Describe': ['Описание 1', 'Описание 2', 'Описание 3'],
            'Model': ['Model_X', 'Model_Y', 'Model_Z'],
            'Segment': ['Electronics', 'Clothing', 'Electronics'],
            'Price': [100, 50, 150],
            'Qty': [2, 1, 3],
            'Sum': [200, 50, 450]
        }
        st.dataframe(pd.DataFrame(example_data))

if __name__ == "__main__":
    create_dashboard()
