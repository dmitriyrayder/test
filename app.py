import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SalesBasedRecommenderSystem:
    def __init__(self):
        self.processed_data = None
        self.le_magazin = LabelEncoder()
        self.le_art = LabelEncoder()
        self.segment_stats = None
        self.item_stats = None
        
    def process_datasales(self, df):
        """Обработка колонки Datasales с различными вариантами"""
        if 'Datasales' not in df.columns:
            return df
        
        datasales_col = df['Datasales'].copy()
        non_null_dates = datasales_col.dropna()
        
        if len(non_null_dates) == 0:
            return df
        
        # Автоматическое определение формата даты
        try:
            parsed_dates = pd.to_datetime(datasales_col, errors='coerce')
            if parsed_dates.notna().sum() > len(non_null_dates) * 0.8:
                df['Datasales'] = parsed_dates.astype('datetime64[ns]')
                df['Month'] = df['Datasales'].dt.month
                df['Quarter'] = df['Datasales'].dt.quarter
                df['Year'] = df['Datasales'].dt.year
                st.info("✅ Колонка Datasales обработана")
        except:
            st.warning("⚠️ Не удалось обработать колонку Datasales")
        
        return df
    
    def process_data(self, df, selected_segment=None):
        """Предобработка данных с фильтрацией по сегменту"""
        df = df.copy()
        
        # Обработка дат
        df = self.process_datasales(df)
        
        # Базовая очистка
        df = df.dropna(subset=['Magazin', 'Art', 'Price', 'Qty', 'Segment'])
        df = df[df['Price'] > 0]
        df = df[df['Qty'] > 0]
        
        # Приведение к строковому типу
        df['Magazin'] = df['Magazin'].astype(str)
        df['Art'] = df['Art'].astype(str)
        df['Segment'] = df['Segment'].astype(str)
        df['Model'] = df['Model'].astype(str)
        
        # Фильтрация по сегменту
        if selected_segment and selected_segment != 'Все':
            df = df[df['Segment'] == selected_segment]
            
        if len(df) == 0:
            return None
        
        # Создание статистики по товарам
        self.item_stats = df.groupby('Art').agg({
            'Qty': ['sum', 'count'],  # общее количество и количество транзакций
            'Magazin': 'nunique',    # количество уникальных магазинов
            'Price': 'mean',         # средняя цена
            'Segment': 'first',      # сегмент
            'Model': 'first'         # модель
        }).round(2)
        
        # Упрощение названий колонок
        self.item_stats.columns = ['total_qty', 'transactions', 'stores', 'avg_price', 'segment', 'model']
        self.item_stats.reset_index(inplace=True)
        
        # Агрегация по магазин-товар
        agg_data = df.groupby(['Magazin', 'Art']).agg({
            'Qty': 'sum',
            'Price': 'mean',
            'Segment': 'first',
            'Model': 'first'
        }).reset_index()
        
        # Кодирование
        agg_data['magazin_encoded'] = self.le_magazin.fit_transform(agg_data['Magazin'])
        agg_data['art_encoded'] = self.le_art.fit_transform(agg_data['Art'])
        
        self.processed_data = agg_data
        return agg_data
    
    def calculate_segment_statistics(self, df):
        """Расчет статистики по сегментам"""
        segment_stats = df.groupby('Segment').agg({
            'Art': 'nunique',
            'Magazin': 'nunique', 
            'Qty': ['sum', 'mean'],
            'Price': 'mean'
        }).round(2)
        
        segment_stats.columns = ['unique_items', 'unique_stores', 'total_qty', 'avg_qty_per_transaction', 'avg_price']
        segment_stats.reset_index(inplace=True)
        segment_stats = segment_stats.sort_values('total_qty', ascending=False)
        
        self.segment_stats = segment_stats
        return segment_stats
    
    def get_recommendations_by_sales(self, magazin_name, top_k=10, min_transactions=2):
        """Рекомендации на основе штучных продаж"""
        if self.processed_data is None or self.item_stats is None:
            return None
        
        try:
            # Получение товаров, которые магазин уже покупал
            magazin_items = set(self.processed_data[
                self.processed_data['Magazin'] == magazin_name
            ]['Art'].values)
            
            # Фильтрация товаров по минимальному количеству транзакций
            eligible_items = self.item_stats[
                (self.item_stats['transactions'] >= min_transactions) &
                (~self.item_stats['Art'].isin(magazin_items))
            ].copy()
            
            if len(eligible_items) == 0:
                return []
            
            # Сортировка по общему количеству штучных продаж
            eligible_items = eligible_items.sort_values('total_qty', ascending=False)
            
            # Выбор топ-K товаров
            top_items = eligible_items.head(top_k).reset_index(drop=True)
            
            # Формирование рекомендаций
            recommendations = []
            for idx, row in top_items.iterrows():
                recommendations.append({
                    'rank': idx + 1,
                    'item': row['Art'],
                    'total_qty': int(row['total_qty']),
                    'transactions': int(row['transactions']),
                    'stores': int(row['stores']),
                    'avg_price': row['avg_price'],
                    'segment': row['segment'],
                    'model': row['model']
                })
            
            return recommendations
            
        except Exception as e:
            st.error(f"Ошибка при генерации рекомендаций: {str(e)}")
            return None
    
    def get_top_items_statistics(self, top_n=20):
        """Получение статистики по топ товарам"""
        if self.item_stats is None:
            return None
        
        return self.item_stats.sort_values('total_qty', ascending=False).head(top_n)
    
    def get_store_statistics(self, magazin_name):
        """Получение статистики по конкретному магазину"""
        if self.processed_data is None:
            return None
        
        store_data = self.processed_data[self.processed_data['Magazin'] == magazin_name]
        
        if len(store_data) == 0:
            return None
        
        stats = {
            'total_items': len(store_data),
            'total_qty': store_data['Qty'].sum(),
            'avg_qty_per_item': store_data['Qty'].mean(),
            'segments': store_data['Segment'].nunique(),
            'top_segment': store_data.groupby('Segment')['Qty'].sum().idxmax(),
            'avg_price': store_data['Price'].mean()
        }
        
        return stats

def create_dashboard():
    st.set_page_config(page_title="Рекомендательная система", layout="wide")
    
    st.title("🛍️ Рекомендательная система на основе штучных продаж")
    st.markdown("*Рекомендации товаров с наибольшим количеством штучных продаж*")
    st.markdown("---")
    
    # Инициализация системы
    if 'recommender' not in st.session_state:
        st.session_state.recommender = SalesBasedRecommenderSystem()
    
    # Боковая панель
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Выберите Excel файл", 
        type=['xlsx', 'xls'],
        help="Файл должен содержать: Magazin, Art, Segment, Model, Price, Qty"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            
            # Предварительная обработка типов
            if 'Price' in df.columns:
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            if 'Qty' in df.columns:
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
            
            df = df.dropna(subset=['Price', 'Qty'])
            
            # Проверка колонок
            required_cols = ['Magazin', 'Art', 'Segment', 'Model', 'Price', 'Qty']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
                return
            
            if len(df) == 0:
                st.error("Нет валидных данных после очистки")
                return
            
            # Фильтр по сегменту
            st.sidebar.header("🎯 Фильтры")
            segments = ['Все'] + sorted(df['Segment'].unique().tolist())
            selected_segment = st.sidebar.selectbox("Выберите сегмент:", segments)
            
            # Настройки рекомендаций
            st.sidebar.header("⚙️ Настройки")
            min_transactions = st.sidebar.number_input(
                "Минимум транзакций:",
                min_value=1, max_value=50, value=2
            )
            
            # Обработка данных
            processed_df = st.session_state.recommender.process_data(df, selected_segment)
            
            if processed_df is None:
                st.error(f"Нет данных для сегмента '{selected_segment}'")
                return
            
            # Расчет статистики по сегментам
            segment_stats = st.session_state.recommender.calculate_segment_statistics(df)
            
            # Отображение основной информации
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Записей", len(processed_df))
            with col2:
                st.metric("Магазинов", processed_df['Magazin'].nunique())
            with col3:
                st.metric("Товаров", processed_df['Art'].nunique())
            with col4:
                current_segment = selected_segment if selected_segment != 'Все' else 'Все сегменты'
                st.metric("Сегмент", current_segment)
            
            # Основные разделы
            st.markdown("---")
            
            # Табы для разных разделов
            tab1, tab2, tab3 = st.tabs(["📊 Статистика", "🎯 Рекомендации", "📈 Топ товары"])
            
            with tab1:
                st.header("📊 Общая статистика")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Статистика по сегментам")
                    if segment_stats is not None:
                        # Переименование колонок для отображения
                        display_segment_stats = segment_stats.rename(columns={
                            'Segment': 'Сегмент',
                            'unique_items': 'Уникальных товаров',
                            'unique_stores': 'Магазинов',
                            'total_qty': 'Общее количество',
                            'avg_qty_per_transaction': 'Среднее за транзакцию',
                            'avg_price': 'Средняя цена'
                        })
                        st.dataframe(display_segment_stats, use_container_width=True)
                
                with col2:
                    st.subheader("Топ товары по продажам")
                    top_items = st.session_state.recommender.get_top_items_statistics(10)
                    if top_items is not None:
                        display_top_items = top_items[['Art', 'total_qty', 'transactions', 'stores', 'segment']].rename(columns={
                            'Art': 'Товар',
                            'total_qty': 'Общее количество',
                            'transactions': 'Транзакций',
                            'stores': 'Магазинов',
                            'segment': 'Сегмент'
                        })
                        st.dataframe(display_top_items, use_container_width=True)
                
                # Дополнительная статистика
                st.subheader("📋 Детальная статистика")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_qty = df['Qty'].sum() if selected_segment == 'Все' else df[df['Segment'] == selected_segment]['Qty'].sum()
                    st.metric("Общее количество продаж", f"{total_qty:,}")
                
                with col2:
                    avg_transaction = df['Qty'].mean() if selected_segment == 'Все' else df[df['Segment'] == selected_segment]['Qty'].mean()
                    st.metric("Среднее за транзакцию", f"{avg_transaction:.2f}")
                
                with col3:
                    current_data = df if selected_segment == 'Все' else df[df['Segment'] == selected_segment]
                    unique_pairs = len(current_data.groupby(['Magazin', 'Art']).size())
                    st.metric("Уникальных пар магазин-товар", unique_pairs)
                
                with col4:
                    avg_price = df['Price'].mean() if selected_segment == 'Все' else df[df['Segment'] == selected_segment]['Price'].mean()
                    st.metric("Средняя цена", f"{avg_price:.2f}")
            
            with tab2:
                st.header("🎯 Рекомендации для магазина")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_store = st.selectbox(
                        "Выберите магазин:",
                        options=st.session_state.recommender.le_magazin.classes_
                        if hasattr(st.session_state.recommender, 'le_magazin') and 
                           hasattr(st.session_state.recommender.le_magazin, 'classes_')
                        else processed_df['Magazin'].unique()
                    )
                with col2:
                    top_k = st.slider("Количество рекомендаций:", 5, 20, 10)
                
                if st.button("🚀 Получить рекомендации", type="primary"):
                    recommendations = st.session_state.recommender.get_recommendations_by_sales(
                        selected_store, top_k, min_transactions
                    )
                    
                    if recommendations:
                        # Отображение рекомендаций
                        st.subheader(f"Рекомендации для магазина: {selected_store}")
                        
                        rec_df = pd.DataFrame(recommendations)
                        display_rec_df = rec_df.rename(columns={
                            'rank': 'Ранг',
                            'item': 'Товар',
                            'total_qty': 'Общие продажи',
                            'transactions': 'Транзакций',
                            'stores': 'Магазинов',
                            'avg_price': 'Средняя цена',
                            'segment': 'Сегмент',
                            'model': 'Модель'
                        })
                        
                        st.dataframe(display_rec_df, use_container_width=True)
                        
                        # Статистика рекомендаций
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_potential_sales = rec_df['total_qty'].sum()
                            st.metric("Потенциал продаж", f"{total_potential_sales:,}")
                        with col2:
                            avg_transactions = rec_df['transactions'].mean()
                            st.metric("Среднее транзакций", f"{avg_transactions:.1f}")
                        with col3:
                            avg_stores = rec_df['stores'].mean()
                            st.metric("Среднее магазинов", f"{avg_stores:.1f}")
                        with col4:
                            avg_price = rec_df['avg_price'].mean()
                            st.metric("Средняя цена", f"{avg_price:.2f}")
                        
                        # Статистика текущего магазина
                        st.subheader("📊 Статистика выбранного магазина")
                        store_stats = st.session_state.recommender.get_store_statistics(selected_store)
                        
                        if store_stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Товаров в ассортименте", store_stats['total_items'])
                            with col2:
                                st.metric("Общие продажи", f"{store_stats['total_qty']:,}")
                            with col3:
                                st.metric("Количество сегментов", store_stats['segments'])
                            with col4:
                                st.metric("Топ сегмент", store_stats['top_segment'])
                    
                    else:
                        st.info("Нет подходящих рекомендаций для данного магазина")
            
            with tab3:
                st.header("📈 Анализ топ товаров")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    top_n_items = st.slider("Количество товаров для анализа:", 10, 50, 20)
                
                top_items_full = st.session_state.recommender.get_top_items_statistics(top_n_items)
                
                if top_items_full is not None:
                    # Полная таблица топ товаров
                    st.subheader(f"Топ {top_n_items} товаров по штучным продажам")
                    
                    display_items = top_items_full.rename(columns={
                        'Art': 'Товар',
                        'total_qty': 'Общие продажи',
                        'transactions': 'Транзакций',
                        'stores': 'Магазинов', 
                        'avg_price': 'Средняя цена',
                        'segment': 'Сегмент',
                        'model': 'Модель'
                    })
                    
                    st.dataframe(display_items, use_container_width=True)
                    
                    # Анализ топ товаров
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("По сегментам")
                        segment_analysis = top_items_full.groupby('segment')['total_qty'].sum().sort_values(ascending=False)
                        st.bar_chart(segment_analysis)
                    
                    with col2:
                        st.subheader("Распределение транзакций")
                        # Создание групп по количеству транзакций
                        bins = [0, 5, 10, 20, 50, float('inf')]
                        labels = ['1-5', '6-10', '11-20', '21-50', '50+']
                        top_items_full['transaction_group'] = pd.cut(
                            top_items_full['transactions'], 
                            bins=bins, 
                            labels=labels, 
                            right=False
                        )
                        transaction_dist = top_items_full['transaction_group'].value_counts()
                        st.bar_chart(transaction_dist)
                    
                    with col3:
                        st.subheader("Ключевые метрики")
                        st.metric("Общие продажи топ товаров", f"{top_items_full['total_qty'].sum():,}")
                        st.metric("Средние продажи на товар", f"{top_items_full['total_qty'].mean():.0f}")
                        st.metric("Средняя цена", f"{top_items_full['avg_price'].mean():.2f}")
                        st.metric("Лидирующий сегмент", top_items_full.groupby('segment')['total_qty'].sum().idxmax())
                
                # Возможность скачать результаты
                if top_items_full is not None:
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                    
                    csv_data = convert_df_to_csv(display_items)
                    st.download_button(
                        label="📥 Скачать топ товары (CSV)",
                        data=csv_data,
                        file_name=f'top_items_{selected_segment.lower().replace(" ", "_")}.csv',
                        mime='text/csv'
                    )
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            st.error("Проверьте формат и содержимое файла")
    
    else:
        # Инструкции для пользователя
        st.info("👆 Загрузите Excel файл для начала работы")
        
        st.markdown("### 📋 Требуемые колонки:")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("- **Magazin** - название магазина")
            st.markdown("- **Art** - код/название товара") 
            st.markdown("- **Segment** - сегмент товара")
        with cols[1]:
            st.markdown("- **Model** - модель товара")
            st.markdown("- **Price** - цена")
            st.markdown("- **Qty** - количество (штуки)")
        
        st.markdown("### 🎯 Возможности системы:")
        st.markdown("- **Анализ по сегментам** - детальная статистика по каждому сегменту")
        st.markdown("- **Рекомендации по продажам** - товары с наибольшими штучными продажами")
        st.markdown("- **Фильтрация по популярности** - исключение товаров с малым количеством транзакций")
        st.markdown("- **Табличная статистика** - подробная аналитика по товарам и магазинам")

if __name__ == "__main__":
    create_dashboard()
