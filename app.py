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
    """ABC анализ для сегмента - ИСПРАВЛЕНО"""
    segment_data = df[df['Segment'] == segment]
    product_revenue = segment_data.groupby('Art')['Sum'].sum().sort_values(ascending=False)
    
    if product_revenue.empty or product_revenue.sum() == 0:
        return pd.DataFrame(columns=['Art', 'Revenue', 'ABC'])
    
    total_revenue = product_revenue.sum()
    cumulative_revenue = product_revenue.cumsum()
    cumulative_percentage = (cumulative_revenue / total_revenue) * 100
    
    # ИСПРАВЛЕНО: Правильное присвоение категорий ABC
    abc_categories = pd.cut(cumulative_percentage, 
                           bins=[0, 80, 95, 100], 
                           labels=['A', 'B', 'C'],
                           include_lowest=True)
    
    abc_df = pd.DataFrame({
        'Art': product_revenue.index,
        'Revenue': product_revenue.values,
        'ABC': abc_categories
    })
    return abc_df

def analyze_product_lifecycle(df, segment):
    """Анализ жизненного цикла товаров - ИСПРАВЛЕНО"""
    segment_data = df[df['Segment'] == segment]
    
    lifecycle_data = []
    
    for art in segment_data['Art'].unique():
        product_data = segment_data[segment_data['Art'] == art]
        monthly_sales = product_data.groupby('Month')['Qty'].sum()
        
        if len(monthly_sales) == 0:
            continue
            
        total_sales = monthly_sales.sum()
        months_active = len(monthly_sales[monthly_sales > 0])
        
        # ИСПРАВЛЕНО: Безопасная проверка длины данных
        if months_active <= 2:
            stage = 'Внедрение'
        elif len(monthly_sales) >= 6:  # ИСПРАВЛЕНО: проверка достаточности данных
            recent_avg = monthly_sales.iloc[-3:].mean()
            early_avg = monthly_sales.iloc[:3].mean()
            if recent_avg > early_avg * 1.1:
                stage = 'Рост'
            elif monthly_sales.std() < monthly_sales.mean() * 0.3:
                stage = 'Зрелость'
            else:
                stage = 'Спад'
        else:
            # Если данных мало, определяем по тренду
            if monthly_sales.is_monotonic_increasing:
                stage = 'Рост'
            else:
                stage = 'Зрелость'
        
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
    """Генерация алертов и уведомлений - ИСПРАВЛЕНО"""
    alerts = []
    
    # Алерт 1: Товары с резким падением продаж - ИСПРАВЛЕНО
    if df['Datasales'].notna().any():
        recent_data = df[df['Datasales'] >= df['Datasales'].max() - pd.Timedelta(days=30)]
        store_data = recent_data[(recent_data['Magazin'] == store) & (recent_data['Segment'] == segment)]
        
        if not store_data.empty:
            # ИСПРАВЛЕНО: Сравниваем суммы за период, а не средние
            recent_sales = store_data.groupby('Art')['Qty'].sum()
            
            # Берем данные за предыдущий месяц для сравнения
            previous_period_start = df['Datasales'].max() - pd.Timedelta(days=60)
            previous_period_end = df['Datasales'].max() - pd.Timedelta(days=30)
            previous_data = df[(df['Datasales'] >= previous_period_start) & 
                              (df['Datasales'] < previous_period_end) &
                              (df['Magazin'] == store) & 
                              (df['Segment'] == segment)]
            previous_sales = previous_data.groupby('Art')['Qty'].sum()
            
            for art in recent_sales.index:
                if art in previous_sales.index and previous_sales[art] > 0:
                    decline_pct = (previous_sales[art] - recent_sales[art]) / previous_sales[art]
                    if decline_pct > 0.5:  # Падение более чем на 50%
                        product_name = df[df['Art'] == art]['Describe'].iloc[0]
                        alerts.append({
                            'type': 'warning',
                            'title': 'Падение продаж',
                            'message': f'Товар "{product_name}" ({art}) показывает падение продаж на {decline_pct*100:.0f}%',
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
    
    # Алерт 3: Критически мало товаров в ассортименте - ИСПРАВЛЕНО
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    
    # ИСПРАВЛЕНО: Защита от деления на ноль
    if segment_unique > 0:
        coverage = (store_unique / segment_unique * 100)
        if coverage < 20:
            alerts.append({
                'type': 'error',
                'title': 'Критически низкое покрытие',
                'message': f'Покрытие ассортимента составляет только {coverage:.1f}%',
                'priority': 'high'
            })
    
    return alerts

def calculate_seasonality(df, segment):
    """Анализ сезонности для сегмента - ИСПРАВЛЕНО"""
    segment_data = df[df['Segment'] == segment]
    monthly_sales = segment_data.groupby('Month')['Qty'].sum().reindex(range(1, 13), fill_value=0)
    
    # ИСПРАВЛЕНО: Защита от пустых данных
    if monthly_sales.sum() == 0:
        month_names = {1:'Янв', 2:'Фев', 3:'Мар', 4:'Апр', 5:'Май', 6:'Июн',
                       7:'Июл', 8:'Авг', 9:'Сен', 10:'Окт', 11:'Ноя', 12:'Дек'}
        return {
            'months': [month_names[i] for i in range(1, 13)],
            'sales': monthly_sales.values,
            'peak_month': 'Нет данных',
            'low_month': 'Нет данных'
        }
    
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
    
    # Статистика по магазину
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    store_stats = store_data.groupby('Art')['Qty'].sum().reset_index()
    store_stats.columns = ['Art', 'Store_Qty']
    
    # Объединение данных
    merged = network_stats.merge(store_stats, on='Art', how='left')
    merged['Store_Qty'] = merged['Store_Qty'].fillna(0)
    
    # Фильтрация
    filtered = merged[
        (merged['Total_Qty'] >= min_network_qty) &
        (merged['Store_Qty'] <= max_store_qty)
    ]
    
    # ABC категории
    abc_df = calculate_abc_analysis(df, segment)
    filtered = filtered.merge(abc_df[['Art', 'ABC']], on='Art', how='left')
    
    # Расчет потенциала
    filtered['Potential_Qty'] = (filtered['Total_Qty'] / filtered['Store_Count']).round(0)
    filtered['Potential_Sum'] = filtered['Potential_Qty'] * filtered['Avg_Price']
    
    # Сортировка
    filtered = filtered.sort_values('Potential_Qty', ascending=False)
    
    return filtered

def create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df):
    """Создание таблицы статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    stats = []
    
    # Общая статистика
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Всего товаров в сегменте', 
                  'Значение': segment_data['Art'].nunique()})
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Товаров в магазине', 
                  'Значение': store_data['Art'].nunique()})
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Покрытие ассортимента', 
                  'Значение': f"{(store_data['Art'].nunique() / segment_data['Art'].nunique() * 100):.1f}%" if segment_data['Art'].nunique() > 0 else "0%"})
    
    # ABC статистика
    if not abc_df.empty:
        for category in ['A', 'B', 'C']:
            count = len(abc_df[abc_df['ABC'] == category])
            stats.append({'Категория': 'ABC Анализ', 'Показатель': f'Товары категории {category}', 
                         'Значение': count})
    
    # Жизненный цикл
    if not lifecycle_df.empty:
        for stage in ['Внедрение', 'Рост', 'Зрелость', 'Спад']:
            count = len(lifecycle_df[lifecycle_df['Stage'] == stage])
            stats.append({'Категория': 'Жизненный цикл', 'Показатель': stage, 
                         'Значение': count})
    
    # Рекомендации
    if not recommendations.empty:
        stats.append({'Категория': 'Рекомендации', 'Показатель': 'Товаров рекомендовано', 
                     'Значение': len(recommendations)})
        stats.append({'Категория': 'Рекомендации', 'Показатель': 'Потенциал продаж (шт)', 
                     'Значение': int(recommendations['Potential_Qty'].sum())})
        stats.append({'Категория': 'Рекомендации', 'Показатель': 'Потенциальная выручка (грн)', 
                     'Значение': f"{recommendations['Potential_Sum'].sum():,.0f}"})
    
    return pd.DataFrame(stats)

def create_excel_report(df, store, segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts):
    """Создание Excel отчета"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Рекомендации
        if not recommendations.empty:
            recommendations.to_excel(writer, sheet_name='Рекомендации', index=False)
        
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
        
        # ИСПРАВЛЕНО: Цветовое выделение ABC - используем map вместо applymap
        def highlight_abc(val):
            colors = {'A': 'background-color: #90EE90', 'B': 'background-color: #FFE4B5', 'C': 'background-color: #FFB6C1'}
            return colors.get(val, '')
        
        styled_df = display_df.style.map(highlight_abc, subset=['ABC'])
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
        if category != '':
            st.write(f"**{category}**")
            category_data = stats_table[stats_table['Категория'] == category]
            category_display = category_data[['Показатель', 'Значение']].copy()
            st.dataframe(category_display, use_container_width=True, hide_index=True)
            st.write("")
    
    # Жизненный цикл товаров
    st.subheader("🔄 Анализ жизненного цикла товаров")
    if not lifecycle_df.empty:
        stage_summary = lifecycle_df['Stage'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        stages = ['Внедрение', 'Рост', 'Зрелость', 'Спад']
        icons = ['🚀', '📈', '⚖️', '📉']
        
        for i, (stage, icon) in enumerate(zip(stages, icons)):
            with [col1, col2, col3, col4][i]:
                st.metric(f"{icon} {stage}", stage_summary.get(stage, 0))
        
        lifecycle_display = lifecycle_df[['Art', 'Describe', 'Stage', 'Total_Sales', 'Months_Active']].copy()
        lifecycle_display.columns = ['Артикул', 'Описание', 'Стадия', 'Всего продаж', 'Месяцев активности']
        st.dataframe(lifecycle_display, use_container_width=True)
        
        fig_lifecycle = px.pie(values=stage_summary.values, names=stage_summary.index,
                              title="Распределение товаров по стадиям жизненного цикла")
        st.plotly_chart(fig_lifecycle, use_container_width=True)
    
    # Сезонность
    st.subheader("📅 Анализ сезонности")
    season_fig = px.line(x=seasonality_data['months'], y=seasonality_data['sales'],
                        title=f"Сезонность продаж в сегменте {segment}",
                        labels={'x': 'Месяц', 'y': 'Количество продаж'})
    if seasonality_data['peak_month'] != 'Нет данных':
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
            recommendations = generate_recommendations_with_abc(df, selected_store, selected_segment, min_network_qty, max_store_qty)
            abc_df = calculate_abc_analysis(df, selected_segment)
            seasonality_data = calculate_seasonality(df, selected_segment)
            lifecycle_df = analyze_product_lifecycle(df, selected_segment)
            alerts = generate_alerts(df, selected_store, selected_segment, recommendations)
            
            st.subheader("📈 Результаты анализа")
            display_results(df, selected_store, selected_segment, recommendations, seasonality_data, lifecycle_df, alerts, abc_df)
            
            excel_report = create_excel_report(df, selected_store, selected_segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts)
            st.download_button(
                label="📊 Скачать полный отчет Excel",
                data=excel_report.getvalue(),
                file_name=f"analysis_report_{selected_store}_{selected_segment}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
