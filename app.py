import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="Рекомендательная система товаров",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_data
def load_and_process_data(uploaded_file):
    """Загрузка и предобработка данных из Excel"""
    try:
        # Загрузка данных
        df = pd.read_excel(uploaded_file)
        
        # Проверка наличия обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Отсутствуют колонки: {missing_cols}")
            return None
        
        # Обработка дат с автоопределением формата
        df['Datasales'] = parse_dates(df['Datasales'])
        
        # Очистка данных
        df = df.dropna(subset=['Art', 'Magazin', 'Segment'])
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['Sum'] = pd.to_numeric(df['Sum'], errors='coerce').fillna(0)
        
        # Удаление дубликатов
        df = df.drop_duplicates()
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

def parse_dates(date_series):
    """Автоматическое определение и парсинг формата дат"""
    date_formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%y', '%d/%m/%y']
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_series, format=fmt, errors='coerce')
        except:
            continue
    
    # Если стандартные форматы не подошли
    return pd.to_datetime(date_series, infer_datetime_format=True, errors='coerce')

def calculate_network_stats(df, segment):
    """Расчет статистики продаж по сети для сегмента"""
    segment_data = df[df['Segment'] == segment].copy()
    
    network_stats = segment_data.groupby('Art').agg({
        'Qty': ['sum', 'count'],
        'Sum': 'sum',
        'Price': 'mean',
        'Describe': 'first',
        'Model': 'first',
        'Magazin': 'nunique'
    }).reset_index()
    
    # Упрощение названий колонок
    network_stats.columns = ['Art', 'Total_Qty', 'Sales_Count', 'Total_Sum', 'Avg_Price', 'Describe', 'Model', 'Store_Count']
    
    return network_stats

def generate_recommendations(df, store, segment, min_network_qty=10, max_store_qty=2):
    """Генерация рекомендаций для магазина и сегмента"""
    
    # Получение статистики по сети
    network_stats = calculate_network_stats(df, segment)
    
    # Фильтрация товаров с хорошими продажами в сети
    good_network_products = network_stats[
        (network_stats['Total_Qty'] >= min_network_qty) & 
        (network_stats['Store_Count'] >= 2)
    ].copy()
    
    # Получение продаж в выбранном магазине
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)].copy()
    store_sales = store_data.groupby('Art')['Qty'].sum().reset_index()
    store_sales.columns = ['Art', 'Store_Qty']
    
    # Объединение данных
    recommendations = good_network_products.merge(store_sales, on='Art', how='left')
    recommendations['Store_Qty'] = recommendations['Store_Qty'].fillna(0)
    
    # Фильтрация товаров с низкими продажами в магазине
    recommendations = recommendations[recommendations['Store_Qty'] <= max_store_qty].copy()
    
    # Расчет потенциала и приоритета
    recommendations['Potential_Qty'] = recommendations['Total_Qty'] - recommendations['Store_Qty']
    recommendations['Potential_Sum'] = recommendations['Potential_Qty'] * recommendations['Avg_Price']
    recommendations['Priority_Score'] = (
        recommendations['Potential_Qty'] * 0.4 + 
        recommendations['Store_Count'] * 0.3 + 
        (recommendations['Total_Sum'] / recommendations['Total_Sum'].max()) * 100 * 0.3
    )
    
    # Сортировка по приоритету
    recommendations = recommendations.sort_values('Priority_Score', ascending=False)
    
    return recommendations

def display_statistics(df, store, segment):
    """Отображение общей статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    total_products = segment_data['Art'].nunique()
    store_products = store_data['Art'].nunique()
    coverage = (store_products / total_products * 100) if total_products > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Товаров в сегменте", total_products)
    with col2:
        st.metric("Товаров в магазине", store_products)
    with col3:
        st.metric("Покрытие ассортимента", f"{coverage:.1f}%")
    with col4:
        avg_network_qty = segment_data.groupby('Art')['Qty'].sum().mean()
        st.metric("Средние продажи в сети", f"{avg_network_qty:.1f} шт")

def display_recommendations_table(recommendations):
    """Отображение таблицы рекомендаций"""
    if recommendations.empty:
        st.warning("Рекомендации не найдены с текущими параметрами")
        return
    
    # Подготовка данных для отображения
    display_df = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 'Store_Qty', 'Potential_Qty', 'Store_Count']].copy()
    display_df.columns = ['Артикул', 'Описание', 'Модель', 'Цена', 'Продажи сети', 'Продажи магазина', 'Потенциал', 'Магазинов']
    
    # Форматирование числовых колонок
    display_df['Цена'] = display_df['Цена'].round(2)
    display_df['Продажи сети'] = display_df['Продажи сети'].astype(int)
    display_df['Продажи магазина'] = display_df['Продажи магазина'].astype(int)
    display_df['Потенциал'] = display_df['Потенциал'].astype(int)
    display_df['Магазинов'] = display_df['Магазинов'].astype(int)
    
    st.dataframe(display_df, use_container_width=True, height=400)

def display_top_network_products(df, segment, limit=10):
    """Отображение топ товаров сети в сегменте"""
    network_stats = calculate_network_stats(df, segment)
    top_products = network_stats.nlargest(limit, 'Total_Qty')[['Art', 'Describe', 'Total_Qty', 'Avg_Price', 'Store_Count']]
    
    top_products.columns = ['Артикул', 'Описание', 'Продажи', 'Цена', 'Магазинов']
    top_products['Продажи'] = top_products['Продажи'].astype(int)
    top_products['Цена'] = top_products['Цена'].round(2)
    top_products['Магазинов'] = top_products['Магазинов'].astype(int)
    
    st.subheader(f"Топ-{limit} товаров сети в сегменте")
    st.dataframe(top_products, use_container_width=True)

def main():
    st.title("🛍️ Рекомендательная система товаров")
    st.markdown("Система анализирует продажи и рекомендует товары с высоким потенциалом для магазина")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Загрузите Excel файл с данными о продажах", 
        type=['xlsx', 'xls'],
        help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
    )
    
    if uploaded_file is None:
        st.info("👆 Загрузите Excel файл для начала работы")
        return
    
    # Загрузка и обработка данных
    with st.spinner("Загрузка и обработка данных..."):
        df = load_and_process_data(uploaded_file)
    
    if df is None:
        return
    
    st.success(f"✅ Данные загружены: {len(df)} записей")
    
    # Боковая панель с параметрами
    with st.sidebar:
        st.header("⚙️ Параметры")
        
        # Выбор магазина и сегмента
        stores = sorted(df['Magazin'].unique())
        segments = sorted(df['Segment'].unique())
        
        selected_store = st.selectbox("🏪 Выберите магазин:", stores)
        selected_segment = st.selectbox("📊 Выберите сегмент:", segments)
        
        st.subheader("Настройки рекомендаций")
        min_network_qty = st.number_input(
            "Мин. продажи в сети (шт):", 
            min_value=1, max_value=100, value=10,
            help="Минимальное количество продаж товара в сети для рекомендации"
        )
        
        max_store_qty = st.number_input(
            "Макс. продажи в магазине (шт):", 
            min_value=0, max_value=10, value=2,
            help="Максимальное количество продаж в магазине (порог 'разовых продаж')"
        )
        
        # Кнопка генерации рекомендаций
        generate_btn = st.button("🎯 Сделать рекомендацию", type="primary", use_container_width=True)
    
    # Отображение результатов
    if generate_btn:
        with st.spinner("Генерация рекомендаций..."):
            
            # Статистика
            st.subheader("📈 Общая статистика")
            display_statistics(df, selected_store, selected_segment)
            
            # Генерация рекомендаций
            recommendations = generate_recommendations(
                df, selected_store, selected_segment, 
                min_network_qty, max_store_qty
            )
            
            st.subheader("🎯 Рекомендации товаров")
            st.markdown(f"**Магазин:** {selected_store} | **Сегмент:** {selected_segment}")
            
            if not recommendations.empty:
                st.markdown(f"Найдено **{len(recommendations)}** товаров для рекомендации")
                display_recommendations_table(recommendations)
                
                # Дополнительная аналитика
                col1, col2 = st.columns(2)
                
                with col1:
                    total_potential = recommendations['Potential_Qty'].sum()
                    st.metric("Общий потенциал (шт)", int(total_potential))
                
                with col2:
                    potential_revenue = recommendations['Potential_Sum'].sum()
                    st.metric("Потенциальная выручка", f"{potential_revenue:,.0f} руб")
                
                # Топ товаров сети для сравнения
                with st.expander("📊 Топ товары сети в сегменте", expanded=False):
                    display_top_network_products(df, selected_segment)
                
                # Возможность скачать рекомендации
                csv = recommendations.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 Скачать рекомендации (CSV)",
                    data=csv,
                    file_name=f"recommendations_{selected_store}_{selected_segment}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Рекомендации не найдены. Попробуйте изменить параметры фильтрации.")

if __name__ == "__main__":
    main()
