import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(page_title="Рекомендательная система товаров", page_icon="🛍️", layout="wide")

@st.cache_data
def load_and_process_data(uploaded_file):
    """Загрузка и предобработка данных"""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Отсутствуют колонки: {missing_cols}")
            return None
        
        # Обработка дат
        date_formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%y', '%d/%m/%y']
        for fmt in date_formats:
            try:
                df['Datasales'] = pd.to_datetime(df['Datasales'], format=fmt, errors='coerce')
                break
            except:
                continue
        if df['Datasales'].isna().all():
            df['Datasales'] = pd.to_datetime(df['Datasales'], infer_datetime_format=True, errors='coerce')
        
        # Очистка данных
        df = df.dropna(subset=['Art', 'Magazin', 'Segment'])
        for col in ['Qty', 'Price', 'Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df.drop_duplicates()
        df['Month'] = df['Datasales'].dt.month
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

def calculate_abc_analysis(df, segment):
    """ABC анализ для сегмента"""
    segment_data = df[df['Segment'] == segment]
    product_revenue = segment_data.groupby('Art')['Sum'].sum().sort_values(ascending=False)
    
    total_revenue = product_revenue.sum()
    cumulative_revenue = product_revenue.cumsum()
    cumulative_percentage = (cumulative_revenue / total_revenue) * 100
    
    abc_categories = []
    for pct in cumulative_percentage:
        if pct <= 80:
            abc_categories.append('A')
        elif pct <= 95:
            abc_categories.append('B')
        else:
            abc_categories.append('C')
    
    abc_df = pd.DataFrame({
        'Art': product_revenue.index,
        'Revenue': product_revenue.values,
        'ABC': abc_categories
    })
    return abc_df

def calculate_bcg_analysis(df, segment):
    """BCG матрица для сегмента"""
    segment_data = df[df['Segment'] == segment]
    
    # Разделяем данные на два периода для расчета роста
    segment_data = segment_data.sort_values('Datasales')
    mid_date = segment_data['Datasales'].quantile(0.5)
    
    period1 = segment_data[segment_data['Datasales'] <= mid_date]
    period2 = segment_data[segment_data['Datasales'] > mid_date]
    
    sales1 = period1.groupby('Art')['Qty'].sum()
    sales2 = period2.groupby('Art')['Qty'].sum()
    
    # Расчет роста и доли рынка
    bcg_data = []
    total_market = segment_data.groupby('Art')['Qty'].sum().sum()
    
    for art in segment_data['Art'].unique():
        s1 = sales1.get(art, 0)
        s2 = sales2.get(art, 0)
        growth = ((s2 - s1) / s1 * 100) if s1 > 0 else 0
        market_share = segment_data[segment_data['Art'] == art]['Qty'].sum() / total_market * 100
        
        # Определяем категорию BCG
        if growth > 10 and market_share > 5:
            category = 'Звезды'
        elif growth <= 10 and market_share > 5:
            category = 'Дойные коровы'
        elif growth > 10 and market_share <= 5:
            category = 'Знаки вопроса'
        else:
            category = 'Собаки'
        
        bcg_data.append({
            'Art': art,
            'Growth': growth,
            'Market_Share': market_share,
            'BCG_Category': category,
            'Describe': segment_data[segment_data['Art'] == art]['Describe'].iloc[0]
        })
    
    return pd.DataFrame(bcg_data)

def calculate_seasonality(df, segment):
    """Анализ сезонности для сегмента"""
    segment_data = df[df['Segment'] == segment]
    monthly_sales = segment_data.groupby('Month')['Qty'].sum().reindex(range(1, 13), fill_value=0)
    
    # Определяем пиковые и низкие месяцы
    peak_month = monthly_sales.idxmax()
    low_month = monthly_sales.idxmin()
    
    month_names = {1:'Янв', 2:'Фев', 3:'Мар', 4:'Апр', 5:'Май', 6:'Июн',
                   7:'Июл', 8:'Авг', 9:'Сен', 10:'Окт', 11:'Ноя', 12:'Дек'}
    
    seasonality_data = {
        'months': [month_names[i] for i in range(1, 13)],
        'sales': monthly_sales.values,
        'peak_month': month_names[peak_month],
        'low_month': month_names[low_month]
    }
    return seasonality_data

def generate_recommendations_with_abc(df, store, segment, min_network_qty=10, max_store_qty=2):
    """Генерация рекомендаций с ABC анализом"""
    # Статистика по сети
    segment_data = df[df['Segment'] == segment]
    network_stats = segment_data.groupby('Art').agg({
        'Qty': ['sum', 'count'],
        'Sum': 'sum',
        'Price': 'mean',
        'Describe': 'first',
        'Model': 'first',
        'Magazin': 'nunique'
    }).reset_index()
    network_stats.columns = ['Art', 'Total_Qty', 'Sales_Count', 'Total_Sum', 'Avg_Price', 'Describe', 'Model', 'Store_Count']
    
    # ABC анализ
    abc_df = calculate_abc_analysis(df, segment)
    network_stats = network_stats.merge(abc_df[['Art', 'ABC']], on='Art', how='left')
    network_stats['ABC'] = network_stats['ABC'].fillna('C')
    
    # Фильтрация товаров
    good_products = network_stats[
        (network_stats['Total_Qty'] >= min_network_qty) & 
        (network_stats['Store_Count'] >= 2)
    ].copy()
    
    # Продажи в магазине
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    store_sales = store_data.groupby('Art')['Qty'].sum().reset_index()
    store_sales.columns = ['Art', 'Store_Qty']
    
    # Объединение и расчеты
    recommendations = good_products.merge(store_sales, on='Art', how='left')
    recommendations['Store_Qty'] = recommendations['Store_Qty'].fillna(0)
    recommendations = recommendations[recommendations['Store_Qty'] <= max_store_qty].copy()
    
    recommendations['Potential_Qty'] = recommendations['Total_Qty'] - recommendations['Store_Qty']
    recommendations['Potential_Sum'] = recommendations['Potential_Qty'] * recommendations['Avg_Price']
    
    # Приоритет с учетом ABC
    abc_weights = {'A': 1.5, 'B': 1.2, 'C': 1.0}
    recommendations['ABC_Weight'] = recommendations['ABC'].map(abc_weights)
    recommendations['Priority_Score'] = (
        recommendations['Potential_Qty'] * 0.4 + 
        recommendations['Store_Count'] * 0.2 + 
        (recommendations['Total_Sum'] / recommendations['Total_Sum'].max()) * 100 * 0.2 +
        recommendations['ABC_Weight'] * 20
    )
    
    return recommendations.sort_values('Priority_Score', ascending=False)

def create_excel_report(df, store, segment, recommendations, abc_df, bcg_df, seasonality_data):
    """Создание Excel отчета"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Рекомендации
        rec_df = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 'Store_Qty', 
                                'Potential_Qty', 'Store_Count', 'ABC', 'Priority_Score']].copy()
        rec_df.columns = ['Артикул', 'Описание', 'Модель', 'Цена', 'Продажи сети', 'Продажи магазина', 
                         'Потенциал', 'Магазинов', 'ABC', 'Приоритет']
        rec_df.to_excel(writer, sheet_name='Рекомендации', index=False)
        
        # Лист 2: Статистика
        segment_data = df[df['Segment'] == segment]
        store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
        
        stats = pd.DataFrame({
            'Показатель': ['Товаров в сегменте', 'Товаров в магазине', 'Покрытие ассортимента (%)', 
                          'Потенциал продаж (шт)', 'Потенциальная выручка (грн)'],
            'Значение': [
                segment_data['Art'].nunique(),
                store_data['Art'].nunique(),
                round((store_data['Art'].nunique() / segment_data['Art'].nunique() * 100), 1),
                int(recommendations['Potential_Qty'].sum()) if not recommendations.empty else 0,
                round(recommendations['Potential_Sum'].sum(), 0) if not recommendations.empty else 0
            ]
        })
        stats.to_excel(writer, sheet_name='Статистика', index=False)
        
        # Лист 3: BCG матрица
        bcg_df.to_excel(writer, sheet_name='BCG Матрица', index=False)
        
        # Лист 4: Сезонность
        season_df = pd.DataFrame({
            'Месяц': seasonality_data['months'],
            'Продажи': seasonality_data['sales']
        })
        season_df.to_excel(writer, sheet_name='Сезонность', index=False)
    
    output.seek(0)
    return output

def display_results(df, store, segment, recommendations, bcg_df, seasonality_data):
    """Отображение результатов"""
    # Статистика
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Товаров в сегменте", segment_data['Art'].nunique())
    with col2:
        st.metric("Товаров в магазине", store_data['Art'].nunique())
    with col3:
        coverage = (store_data['Art'].nunique() / segment_data['Art'].nunique() * 100) if segment_data['Art'].nunique() > 0 else 0
        st.metric("Покрытие ассортимента", f"{coverage:.1f}%")
    with col4:
        avg_sales = segment_data.groupby('Art')['Qty'].sum().mean()
        st.metric("Средние продажи в сети", f"{avg_sales:.1f} шт")
    
    # Таблица рекомендаций
    st.subheader("🎯 Рекомендации товаров")
    if not recommendations.empty:
        display_df = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 
                                    'Store_Qty', 'Potential_Qty', 'Store_Count', 'ABC']].copy()
        display_df.columns = ['Артикул', 'Описание', 'Модель', 'Цена', 'Продажи сети', 
                             'Продажи магазина', 'Потенциал', 'Магазинов', 'ABC']
        
        # Цветовое выделение ABC
        def highlight_abc(val):
            colors = {'A': 'background-color: #90EE90', 'B': 'background-color: #FFE4B5', 'C': 'background-color: #FFB6C1'}
            return colors.get(val, '')
        
        styled_df = display_df.style.applymap(highlight_abc, subset=['ABC'])
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Общий потенциал (шт)", int(recommendations['Potential_Qty'].sum()))
        with col2:
            st.metric("Потенциальная выручка", f"{recommendations['Potential_Sum'].sum():,.0f} грн")
    else:
        st.info("Рекомендации не найдены. Попробуйте изменить параметры.")
    
    # BCG матрица
    st.subheader("📊 BCG Матрица")
    if not bcg_df.empty:
        fig = px.scatter(bcg_df, x='Market_Share', y='Growth', color='BCG_Category',
                        hover_data=['Art', 'Describe'], title="BCG Матрица товаров",
                        labels={'Market_Share': 'Доля рынка (%)', 'Growth': 'Рост продаж (%)'})
        fig.add_hline(y=10, line_dash="dash", line_color="gray")
        fig.add_vline(x=5, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Сводка BCG
        bcg_summary = bcg_df['BCG_Category'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        categories = ['Звезды', 'Дойные коровы', 'Знаки вопроса', 'Собаки']
        colors = ['🌟', '🐄', '❓', '🐕']
        
        for i, (cat, color) in enumerate(zip(categories, colors)):
            with [col1, col2, col3, col4][i]:
                st.metric(f"{color} {cat}", bcg_summary.get(cat, 0))
    
    # Сезонность
    st.subheader("📅 Анализ сезонности")
    season_fig = px.line(x=seasonality_data['months'], y=seasonality_data['sales'],
                        title=f"Сезонность продаж в сегменте {segment}",
                        labels={'x': 'Месяц', 'y': 'Количество продаж'})
    season_fig.add_annotation(x=seasonality_data['peak_month'], y=max(seasonality_data['sales']),
                             text=f"Пик: {seasonality_data['peak_month']}", showarrow=True, arrowcolor="green")
    st.plotly_chart(season_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📈 **Пиковый месяц:** {seasonality_data['peak_month']}")
    with col2:
        st.info(f"📉 **Низкий месяц:** {seasonality_data['low_month']}")

def main():
    st.title("🛍️ Рекомендательная система товаров")
    st.markdown("Система с ABC/BCG анализом и сезонностью")
    
    uploaded_file = st.file_uploader("Загрузите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is None:
        st.info("👆 Загрузите Excel файл для начала работы")
        return
    
    with st.spinner("Загрузка данных..."):
        df = load_and_process_data(uploaded_file)
    
    if df is None:
        return
    
    st.success(f"✅ Данные загружены: {len(df)} записей")
    
    # Параметры
    with st.sidebar:
        st.header("⚙️ Параметры")
        stores = sorted(df['Magazin'].unique())
        segments = sorted(df['Segment'].unique())
        
        selected_store = st.selectbox("🏪 Магазин:", stores)
        selected_segment = st.selectbox("📊 Сегмент:", segments)
        
        min_network_qty = st.number_input("Мин. продажи в сети:", min_value=1, max_value=100, value=10)
        max_store_qty = st.number_input("Макс. продажи в магазине:", min_value=0, max_value=10, value=2)
        
        analyze_btn = st.button("🎯 Анализировать", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("Анализ данных..."):
            # Расчеты
            recommendations = generate_recommendations_with_abc(df, selected_store, selected_segment, min_network_qty, max_store_qty)
            abc_df = calculate_abc_analysis(df, selected_segment)
            bcg_df = calculate_bcg_analysis(df, selected_segment)
            seasonality_data = calculate_seasonality(df, selected_segment)
            
            # Отображение
            st.subheader("📈 Результаты анализа")
            display_results(df, selected_store, selected_segment, recommendations, bcg_df, seasonality_data)
            
            # Скачивание отчета
            excel_report = create_excel_report(df, selected_store, selected_segment, recommendations, abc_df, bcg_df, seasonality_data)
            st.download_button(
                label="📊 Скачать полный отчет Excel",
                data=excel_report.getvalue(),
                file_name=f"analysis_report_{selected_store}_{selected_segment}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
