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

def analyze_product_lifecycle(df, segment):
    """Анализ жизненного цикла товаров"""
    segment_data = df[df['Segment'] == segment]
    
    # Группируем данные по товарам и месяцам
    lifecycle_data = []
    
    for art in segment_data['Art'].unique():
        product_data = segment_data[segment_data['Art'] == art]
        monthly_sales = product_data.groupby('Month')['Qty'].sum()
        
        # Определяем стадию жизненного цикла
        if len(monthly_sales) == 0:
            continue
            
        total_sales = monthly_sales.sum()
        max_sales = monthly_sales.max()
        months_active = len(monthly_sales[monthly_sales > 0])
        
        # Логика определения стадии
        if months_active <= 2:
            stage = 'Внедрение'
        elif monthly_sales.iloc[-3:].mean() > monthly_sales.iloc[:3].mean():
            stage = 'Рост'
        elif monthly_sales.std() < monthly_sales.mean() * 0.3:
            stage = 'Зрелость'
        else:
            stage = 'Спад'
        
        lifecycle_data.append({
            'Art': art,
            'Describe': product_data['Describe'].iloc[0],
            'Total_Sales': total_sales,
            'Months_Active': months_active,
            'Stage': stage,
            'Avg_Monthly_Sales': total_sales / months_active if months_active > 0 else 0
        })
    
    return pd.DataFrame(lifecycle_data)

def generate_alerts(df, store, segment, recommendations):
    """Генерация алертов и уведомлений"""
    alerts = []
    
    # Алерт 1: Товары с резким падением продаж
    recent_data = df[df['Datasales'] >= df['Datasales'].max() - pd.Timedelta(days=30)]
    store_data = recent_data[(recent_data['Magazin'] == store) & (recent_data['Segment'] == segment)]
    
    if not store_data.empty:
        recent_sales = store_data.groupby('Art')['Qty'].sum()
        all_time_avg = df[(df['Magazin'] == store) & (df['Segment'] == segment)].groupby('Art')['Qty'].mean()
        
        for art in recent_sales.index:
            if art in all_time_avg.index:
                if recent_sales[art] < all_time_avg[art] * 0.5:
                    product_name = df[df['Art'] == art]['Describe'].iloc[0]
                    alerts.append({
                        'type': 'warning',
                        'title': 'Падение продаж',
                        'message': f'Товар "{product_name}" ({art}) показывает падение продаж на 50%+',
                        'priority': 'high'
                    })
    
    # Алерт 2: Новые возможности
    if not recommendations.empty:
        top_opportunities = recommendations.head(3)
        for _, row in top_opportunities.iterrows():
            alerts.append({
                'type': 'success',
                'title': 'Новая возможность',
                'message': f'Товар "{row["Describe"]}" ({row["Art"]}) имеет потенциал {int(row["Potential_Qty"])} продаж',
                'priority': 'medium'
            })
    
    # Алерт 3: Критически мало товаров в ассортименте
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    coverage = (store_data['Art'].nunique() / segment_data['Art'].nunique() * 100) if segment_data['Art'].nunique() > 0 else 0
    
    if coverage < 20:
        alerts.append({
            'type': 'error',
            'title': 'Критически низкое покрытие',
            'message': f'Покрытие ассортимента составляет только {coverage:.1f}%',
            'priority': 'high'
        })
    
    return alerts

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

def create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df):
    """Создание подробной таблицы статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    # Общая статистика
    total_products_segment = segment_data['Art'].nunique()
    products_in_store = store_data['Art'].nunique()
    coverage = (products_in_store / total_products_segment * 100) if total_products_segment > 0 else 0
    
    # ABC статистика
    abc_stats = abc_df['ABC'].value_counts() if not abc_df.empty else pd.Series()
    
    # Потенциал
    total_potential_qty = recommendations['Potential_Qty'].sum() if not recommendations.empty else 0
    total_potential_sum = recommendations['Potential_Sum'].sum() if not recommendations.empty else 0
    
    # Продажи
    total_sales_qty = segment_data['Qty'].sum()
    total_sales_sum = segment_data['Sum'].sum()
    avg_price = segment_data['Price'].mean()
    
    # Жизненный цикл
    lifecycle_stats = lifecycle_df['Stage'].value_counts() if not lifecycle_df.empty else pd.Series()
    
    # Формирование таблицы
    stats_data = []
    
    # Раздел 1: Общие показатели
    stats_data.extend([
        ['Общие показатели', 'Товаров в сегменте', f"{total_products_segment:,}"],
        ['', 'Товаров в магазине', f"{products_in_store:,}"],
        ['', 'Покрытие ассортимента', f"{coverage:.1f}%"],
        ['', 'Средняя цена в сегменте', f"{avg_price:.2f} грн"],
        ['', '', ''],
    ])
    
    # Раздел 2: Продажи
    stats_data.extend([
        ['Продажи', 'Общее количество (сегмент)', f"{total_sales_qty:,} шт"],
        ['', 'Общая выручка (сегмент)', f"{total_sales_sum:,.0f} грн"],
        ['', 'Средние продажи на товар', f"{total_sales_qty/total_products_segment:.1f} шт" if total_products_segment > 0 else "0 шт"],
        ['', '', ''],
    ])
    
    # Раздел 3: ABC анализ
    stats_data.extend([
        ['ABC анализ', 'Категория A (80% выручки)', f"{abc_stats.get('A', 0):,} товаров"],
        ['', 'Категория B (15% выручки)', f"{abc_stats.get('B', 0):,} товаров"],
        ['', 'Категория C (5% выручки)', f"{abc_stats.get('C', 0):,} товаров"],
        ['', '', ''],
    ])
    
    # Раздел 4: Потенциал
    stats_data.extend([
        ['Потенциал', 'Рекомендуемых товаров', f"{len(recommendations):,}"],
        ['', 'Потенциал продаж', f"{total_potential_qty:,.0f} шт"],
        ['', 'Потенциальная выручка', f"{total_potential_sum:,.0f} грн"],
        ['', 'Средний потенциал на товар', f"{total_potential_qty/len(recommendations):.1f} шт" if len(recommendations) > 0 else "0 шт"],
        ['', '', ''],
    ])
    
    # Раздел 5: Жизненный цикл
    stats_data.extend([
        ['Жизненный цикл', 'Стадия "Внедрение"', f"{lifecycle_stats.get('Внедрение', 0):,} товаров"],
        ['', 'Стадия "Рост"', f"{lifecycle_stats.get('Рост', 0):,} товаров"],
        ['', 'Стадия "Зрелость"', f"{lifecycle_stats.get('Зрелость', 0):,} товаров"],
        ['', 'Стадия "Спад"', f"{lifecycle_stats.get('Спад', 0):,} товаров"],
    ])
    
    stats_df = pd.DataFrame(stats_data, columns=['Категория', 'Показатель', 'Значение'])
    return stats_df

def create_excel_report(df, store, segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts):
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
        stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)
        stats_table.to_excel(writer, sheet_name='Статистика', index=False)
        
        # Лист 3: Жизненный цикл товаров
        lifecycle_df.to_excel(writer, sheet_name='Жизненный цикл', index=False)
        
        # Лист 4: Сезонность
        season_df = pd.DataFrame({
            'Месяц': seasonality_data['months'],
            'Продажи': seasonality_data['sales']
        })
        season_df.to_excel(writer, sheet_name='Сезонность', index=False)
        
        # Лист 5: Алерты
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_excel(writer, sheet_name='Алерты', index=False)
    
    output.seek(0)
    return output

def display_alerts(alerts):
    """Отображение алертов"""
    if not alerts:
        return
    
    st.subheader("🚨 Алерты и уведомления")
    
    for alert in alerts:
        if alert['type'] == 'error':
            st.error(f"**{alert['title']}**: {alert['message']}")
        elif alert['type'] == 'warning':
            st.warning(f"**{alert['title']}**: {alert['message']}")
        elif alert['type'] == 'success':
            st.success(f"**{alert['title']}**: {alert['message']}")
        else:
            st.info(f"**{alert['title']}**: {alert['message']}")

def display_results(df, store, segment, recommendations, seasonality_data, lifecycle_df, alerts, abc_df):
    """Отображение результатов"""
    # Алерты
    display_alerts(alerts)
    
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
    
    # Подробная таблица статистики
    st.subheader("📊 Подробная статистика")
    stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)
    
    # Группировка по категориям для лучшего отображения
    categories = stats_table['Категория'].unique()
    
    for category in categories:
        if category != '':  # Пропускаем пустые строки-разделители
            st.write(f"**{category}**")
            category_data = stats_table[stats_table['Категория'] == category]
            category_display = category_data[['Показатель', 'Значение']].copy()
            st.dataframe(category_display, use_container_width=True, hide_index=True)
            st.write("")  # Добавляем пустую строку между категориями
    
    # Жизненный цикл товаров
    st.subheader("🔄 Анализ жизненного цикла товаров")
    if not lifecycle_df.empty:
        # Сводка по стадиям
        stage_summary = lifecycle_df['Stage'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        stages = ['Внедрение', 'Рост', 'Зрелость', 'Спад']
        icons = ['🚀', '📈', '⚖️', '📉']
        
        for i, (stage, icon) in enumerate(zip(stages, icons)):
            with [col1, col2, col3, col4][i]:
                st.metric(f"{icon} {stage}", stage_summary.get(stage, 0))
        
        # Детальная таблица
        lifecycle_display = lifecycle_df[['Art', 'Describe', 'Stage', 'Total_Sales', 'Months_Active']].copy()
        lifecycle_display.columns = ['Артикул', 'Описание', 'Стадия', 'Всего продаж', 'Месяцев активности']
        st.dataframe(lifecycle_display, use_container_width=True)
        
        # График распределения по стадиям
        fig_lifecycle = px.pie(values=stage_summary.values, names=stage_summary.index,
                              title="Распределение товаров по стадиям жизненного цикла")
        st.plotly_chart(fig_lifecycle, use_container_width=True)
    
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
    st.markdown("Система с ABC анализом, алертами и анализом жизненного цикла товаров")
    
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
            seasonality_data = calculate_seasonality(df, selected_segment)
            lifecycle_df = analyze_product_lifecycle(df, selected_segment)
            alerts = generate_alerts(df, selected_store, selected_segment, recommendations)
            
            # Отображение
            st.subheader("📈 Результаты анализа")
            display_results(df, selected_store, selected_segment, recommendations, seasonality_data, lifecycle_df, alerts, abc_df)
            
            # Скачивание отчета
            excel_report = create_excel_report(df, selected_store, selected_segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts)
            st.download_button(
                label="📊 Скачать полный отчет Excel",
                data=excel_report.getvalue(),
                file_name=f"analysis_report_{selected_store}_{selected_segment}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
