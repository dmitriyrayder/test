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

def generate_detailed_statistics(df, store, segment, recommendations):
    """Генерация детальной табличной статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    # Статистика по ABC категориям
    abc_stats = []
    if not recommendations.empty:
        for abc_cat in ['A', 'B', 'C']:
            abc_products = recommendations[recommendations['ABC'] == abc_cat]
            if not abc_products.empty:
                abc_stats.append({
                    'ABC_Категория': abc_cat,
                    'Количество_товаров': len(abc_products),
                    'Потенциал_продаж': int(abc_products['Potential_Qty'].sum()),
                    'Потенциальная_выручка': round(abc_products['Potential_Sum'].sum(), 0),
                    'Средний_приоритет': round(abc_products['Priority_Score'].mean(), 1)
                })
    
    # Статистика по магазинам сети
    network_stats = segment_data.groupby('Magazin').agg({
        'Art': 'nunique',
        'Qty': 'sum',
        'Sum': 'sum'
    }).reset_index()
    network_stats.columns = ['Магазин', 'Уникальных_товаров', 'Общее_количество', 'Общая_выручка']
    network_stats = network_stats.sort_values('Общая_выручка', ascending=False)
    
    # Топ товары по продажам в сегменте
    top_products = segment_data.groupby(['Art', 'Describe']).agg({
        'Qty': 'sum',
        'Sum': 'sum',
        'Magazin': 'nunique'
    }).reset_index()
    top_products.columns = ['Артикул', 'Описание', 'Количество', 'Выручка', 'Магазинов']
    top_products = top_products.sort_values('Выручка', ascending=False).head(15)
    
    # Статистика по месяцам
    monthly_stats = segment_data.groupby('Month').agg({
        'Qty': 'sum',
        'Sum': 'sum',
        'Art': 'nunique'
    }).reset_index()
    
    month_names = {1:'Январь', 2:'Февраль', 3:'Март', 4:'Апрель', 5:'Май', 6:'Июнь',
                   7:'Июль', 8:'Август', 9:'Сентябрь', 10:'Октябрь', 11:'Ноябрь', 12:'Декабрь'}
    monthly_stats['Месяц'] = monthly_stats['Month'].map(month_names)
    monthly_stats = monthly_stats[['Месяц', 'Qty', 'Sum', 'Art']]
    monthly_stats.columns = ['Месяц', 'Количество', 'Выручка', 'Уникальных_товаров']
    
    return {
        'abc_stats': pd.DataFrame(abc_stats),
        'network_stats': network_stats,
        'top_products': top_products,
        'monthly_stats': monthly_stats
    }

def create_excel_report(df, store, segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts, detailed_stats):
    """Создание Excel отчета"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Рекомендации
        rec_df = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 'Store_Qty', 
                                'Potential_Qty', 'Store_Count', 'ABC', 'Priority_Score']].copy()
        rec_df.columns = ['Артикул', 'Описание', 'Модель', 'Цена', 'Продажи сети', 'Продажи магазина', 
                         'Потенциал', 'Магазинов', 'ABC', 'Приоритет']
        rec_df.to_excel(writer, sheet_name='Рекомендации', index=False)
        
        # Лист 2: Общая статистика
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
        stats.to_excel(writer, sheet_name='Общая статистика', index=False)
        
        # Лист 3: ABC статистика
        if not detailed_stats['abc_stats'].empty:
            detailed_stats['abc_stats'].to_excel(writer, sheet_name='ABC статистика', index=False)
        
        # Лист 4: Статистика по сети
        detailed_stats['network_stats'].to_excel(writer, sheet_name='Статистика по сети', index=False)
        
        # Лист 5: Топ товары
        detailed_stats['top_products'].to_excel(writer, sheet_name='Топ товары', index=False)
        
        # Лист 6: Месячная статистика
        detailed_stats['monthly_stats'].to_excel(writer, sheet_name='Месячная статистика', index=False)
        
        # Лист 7: Жизненный цикл товаров
        lifecycle_df.to_excel(writer, sheet_name='Жизненный цикл', index=False)
        
        # Лист 8: Сезонность
        season_df = pd.DataFrame({
            'Месяц': seasonality_data['months'],
            'Продажи': seasonality_data['sales']
        })
        season_df.to_excel(writer, sheet_name='Сезонность', index=False)
        
        # Лист 9: Алерты
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

def display_results(df, store, segment, recommendations, seasonality_data, lifecycle_df, alerts):
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
    
    # Детальная табличная статистика
    st.subheader("📊 Детальная статистика")
    detailed_stats = generate_detailed_statistics(df, store, segment, recommendations)
    
    # Создаем табы для разных таблиц
    tab1, tab2, tab3, tab4 = st.tabs(["ABC Статистика", "Статистика по сети", "Топ товары", "Месячная статистика"])
    
    with tab1:
        st.write("**Статистика по ABC категориям:**")
        if not detailed_stats['abc_stats'].empty:
            st.dataframe(detailed_stats['abc_stats'], use_container_width=True)
        else:
            st.info("Нет данных по ABC категориям")
    
    with tab2:
        st.write("**Статистика по магазинам сети:**")
        st.dataframe(detailed_stats['network_stats'], use_container_width=True)
        
        # Выделяем текущий магазин
        current_store_stats = detailed_stats['network_stats'][detailed_stats['network_stats']['Магазин'] == store]
        if not current_store_stats.empty:
            st.info(f"🏪 **Ваш магазин '{store}'** занимает позицию #{detailed_stats['network_stats'].index[detailed_stats['network_stats']['Магазин'] == store].tolist()[0] + 1} по выручке")
    
    with tab3:
        st.write("**Топ-15 товаров по выручке в сегменте:**")
        st.dataframe(detailed_stats['top_products'], use_container_width=True)
        
        # Проверяем, какие из топ товаров есть в магазине
        store_products = store_data['Art'].unique()
        top_in_store = detailed_stats['top_products'][detailed_stats['top_products']['Артикул'].isin(store_products)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Топ товаров в вашем магазине", len(top_in_store))
        with col2:
            coverage_top = (len(top_in_store) / len(detailed_stats['top_products']) * 100)
            st.metric("Покрытие топ товаров", f"{coverage_top:.1f}%")
    
    with tab4:
        st.write("**Статистика продаж по месяцам:**")
        st.dataframe(detailed_stats['monthly_stats'], use_container_width=True)
        
        # График месячной динамики
        fig_monthly = px.bar(detailed_stats['monthly_stats'], x='Месяц', y='Выручка',
                            title="Динамика выручки по месяцам")
        st.plotly_chart(fig_monthly, use_container_width=True)
    
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
    
    return detailed_stats

def main():
    st.title("🛍️ Рекомендательная система товаров")
    st.markdown("Система с ABC анализом, алертами, A/B тестированием и анализом жизненного цикла")
    
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
            detailed_stats = display_results(df, selected_store, selected_segment, recommendations, seasonality_data, lifecycle_df, alerts)
            
            # Скачивание отчета
            excel_report = create_excel_report(df, selected_store, selected_segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts, detailed_stats)report(df, selected_store, selected_segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts)
            st.download_button(
                label="📊 Скачать полный отчет Excel",
                data=excel_report.getvalue(),
                file_name=f"analysis_report_{selected_store}_{selected_segment}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
