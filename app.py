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

def generate_analysis_comments(df, store, segment, recommendations):
    """Генерация комментариев и анализа результатов"""
    comments = []
    
    # Общая статистика
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    total_products = segment_data['Art'].nunique()
    store_products = store_data['Art'].nunique()
    coverage = (store_products / total_products * 100) if total_products > 0 else 0
    
    comments.append(f"## 📊 Анализ результатов для магазина '{store}' в сегменте '{segment}'")
    comments.append(f"")
    comments.append(f"### Общая статистика:")
    comments.append(f"- **Общее количество товаров в сегменте:** {total_products}")
    comments.append(f"- **Товаров представлено в магазине:** {store_products}")
    comments.append(f"- **Покрытие ассортимента:** {coverage:.1f}%")
    
    if not recommendations.empty:
        # Анализ рекомендаций
        total_potential_qty = recommendations['Potential_Qty'].sum()
        total_potential_revenue = recommendations['Potential_Sum'].sum()
        avg_priority = recommendations['Priority_Score'].mean()
        top_recommendation = recommendations.iloc[0]
        
        comments.append(f"")
        comments.append(f"### Результаты рекомендательной системы:")
        comments.append(f"- **Количество рекомендованных товаров:** {len(recommendations)}")
        comments.append(f"- **Общий потенциал продаж:** {int(total_potential_qty)} штук")
        comments.append(f"- **Потенциальная дополнительная выручка:** {total_potential_revenue:,.0f} грн")
        comments.append(f"- **Средний приоритетный балл:** {avg_priority:.1f}")
        
        comments.append(f"")
        comments.append(f"### Топ-рекомендация:")
        comments.append(f"- **Артикул:** {top_recommendation['Art']}")
        comments.append(f"- **Описание:** {top_recommendation['Describe']}")
        comments.append(f"- **Потенциал продаж:** {int(top_recommendation['Potential_Qty'])} штук")
        comments.append(f"- **Средняя цена:** {top_recommendation['Avg_Price']:.2f} грн")
        comments.append(f"- **Представлен в магазинах:** {int(top_recommendation['Store_Count'])}")
        
        # Категоризация рекомендаций
        high_priority = recommendations[recommendations['Priority_Score'] >= 75]
        medium_priority = recommendations[(recommendations['Priority_Score'] >= 50) & (recommendations['Priority_Score'] < 75)]
        low_priority = recommendations[recommendations['Priority_Score'] < 50]
        
        comments.append(f"")
        comments.append(f"### Приоритетность рекомендаций:")
        comments.append(f"- **Высокий приоритет (≥75 баллов):** {len(high_priority)} товаров")
        comments.append(f"- **Средний приоритет (50-74 балла):** {len(medium_priority)} товаров")
        comments.append(f"- **Низкий приоритет (<50 баллов):** {len(low_priority)} товаров")
        
        # Анализ по ценовым сегментам
        price_ranges = pd.cut(recommendations['Avg_Price'], bins=3, labels=['Низкая', 'Средняя', 'Высокая'])
        price_analysis = price_ranges.value_counts()
        
        comments.append(f"")
        comments.append(f"### Распределение по ценовым сегментам:")
        for price_range, count in price_analysis.items():
            comments.append(f"- **{price_range} ценовая категория:** {count} товаров")
        
        # Рекомендации по внедрению
        comments.append(f"")
        comments.append(f"### 💡 Рекомендации по внедрению:")
        
        if coverage < 30:
            comments.append(f"- **Низкое покрытие ассортимента ({coverage:.1f}%)** - рекомендуется активно расширять ассортимент")
        elif coverage < 60:
            comments.append(f"- **Среднее покрытие ассортимента ({coverage:.1f}%)** - есть хорошие возможности для роста")
        else:
            comments.append(f"- **Высокое покрытие ассортимента ({coverage:.1f}%)** - фокус на оптимизации существующих позиций")
        
        if len(high_priority) > 0:
            comments.append(f"- **Приоритет на {len(high_priority)} товаров высокого приоритета** - они имеют наибольший потенциал")
        
        if total_potential_revenue > 50000:
            comments.append(f"- **Высокий потенциал роста выручки** - внедрение рекомендаций может существенно увеличить продажи")
        
        comments.append(f"- **Постепенное внедрение** - начните с топ-5 позиций и отслеживайте результаты")
        comments.append(f"- **Мониторинг эффективности** - регулярно анализируйте продажи новых позиций")
        
    else:
        comments.append(f"")
        comments.append(f"### ⚠️ Рекомендации не найдены")
        comments.append(f"- Попробуйте изменить параметры фильтрации")
        comments.append(f"- Уменьшите минимальные продажи в сети")
        comments.append(f"- Увеличьте максимальные продажи в магазине")
    
    return "\n".join(comments)

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
                    st.metric("Потенциальная выручка", f"{potential_revenue:,.0f} грн")
                
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
            
            # Генерация и отображение аналитических комментариев
            st.subheader("💬 Аналитические комментарии")
            analysis_comments = generate_analysis_comments(df, selected_store, selected_segment, recommendations)
            st.markdown(analysis_comments)
            
            # Возможность скачать аналитический отчет
            st.download_button(
                label="📄 Скачать аналитический отчет",
                data=analysis_comments,
                file_name=f"analysis_report_{selected_store}_{selected_segment}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
