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
st.set_page_config(page_title="Рекомендательная система товаров, которая предлагает магазину перечень товаров, которые хорошо продаются в сети,но еще не представлены в данном магазине", page_icon="🛍️", layout="wide")

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
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        
        if df['Datasales'].isna().all():
            for fmt in date_formats:
                try:
                    df['Datasales'] = pd.to_datetime(df['Datasales'], format=fmt, errors='coerce')
                    if not df['Datasales'].isna().all():
                        break
                except:
                    continue
        
        # Очистка данных
        df = df.dropna(subset=['Art', 'Magazin', 'Segment', 'Datasales'])
        
        # Конвертация числовых колонок
        for col in ['Qty', 'Price', 'Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Удаление дубликатов и некорректных данных
        df = df[df['Qty'] > 0]  # Только положительные продажи
        df = df[df['Price'] > 0]  # Только положительные цены
        df = df.drop_duplicates()
        
        # Добавление временных признаков
        df['Month'] = df['Datasales'].dt.month
        df['Year'] = df['Datasales'].dt.year
        df['Week'] = df['Datasales'].dt.isocalendar().week
        
        return df
        
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {str(e)}")
        return None

@st.cache_data
def load_data_from_google_sheets(sheet_url):
    """Загрузка данных из публичной Google Sheets таблицы"""
    try:
        # Извлечение ID таблицы из URL
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not match:
            st.error("Неверный формат URL Google Sheets")
            return None

        sheet_id = match.group(1)

        # Формирование URL для экспорта в CSV формате
        export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'

        # Загрузка данных
        df = pd.read_csv(export_url)

        # Проверка обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Отсутствуют колонки: {missing_cols}")
            return None

        # Обработка дат
        date_formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%y', '%d/%m/%y']
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')

        if df['Datasales'].isna().all():
            for fmt in date_formats:
                try:
                    df['Datasales'] = pd.to_datetime(df['Datasales'], format=fmt, errors='coerce')
                    if not df['Datasales'].isna().all():
                        break
                except:
                    continue

        # Очистка данных
        df = df.dropna(subset=['Art', 'Magazin', 'Segment', 'Datasales'])

        # Конвертация числовых колонок
        for col in ['Qty', 'Price', 'Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Удаление дубликатов и некорректных данных
        df = df[df['Qty'] > 0]  # Только положительные продажи
        df = df[df['Price'] > 0]  # Только положительные цены
        df = df.drop_duplicates()

        # Добавление временных признаков
        df['Month'] = df['Datasales'].dt.month
        df['Year'] = df['Datasales'].dt.year
        df['Week'] = df['Datasales'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"Ошибка загрузки данных из Google Sheets: {str(e)}")
        return None

def calculate_abc_analysis(df, segment):
    """ABC анализ для сегмента по методу Парето"""
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        return pd.DataFrame(columns=['Art', 'Revenue', 'ABC', 'Cumulative_Pct'])
    
    # Группировка по артикулу
    product_revenue = segment_data.groupby('Art')['Sum'].sum().sort_values(ascending=False)
    
    if product_revenue.empty or product_revenue.sum() == 0:
        return pd.DataFrame(columns=['Art', 'Revenue', 'ABC', 'Cumulative_Pct'])
    
    # Расчет кумулятивного процента
    total_revenue = product_revenue.sum()
    cumulative_revenue = product_revenue.cumsum()
    cumulative_percentage = (cumulative_revenue / total_revenue) * 100
    
    # Присвоение категорий ABC
    # A: 0-80% выручки, B: 80-95%, C: 95-100%
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
        'ABC': abc_categories,
        'Cumulative_Pct': cumulative_percentage.values
    })
    
    return abc_df

def analyze_product_lifecycle(df, segment):
    """Анализ жизненного цикла товаров"""
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        return pd.DataFrame(columns=['Art', 'Describe', 'Total_Sales', 'Months_Active', 'Stage', 'Avg_Monthly_Sales'])
    
    lifecycle_data = []
    
    for art in segment_data['Art'].unique():
        product_data = segment_data[segment_data['Art'] == art].copy()
        
        # Группировка по месяцам
        monthly_sales = product_data.groupby(['Year', 'Month'])['Qty'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values(['Year', 'Month'])
        
        if len(monthly_sales) == 0:
            continue
        
        total_sales = monthly_sales['Qty'].sum()
        months_active = len(monthly_sales[monthly_sales['Qty'] > 0])
        avg_monthly_sales = total_sales / months_active if months_active > 0 else 0
        
        # Определение стадии жизненного цикла
        if months_active <= 2:
            stage = 'Внедрение'
        elif len(monthly_sales) >= 4:
            # Берем первую и последнюю трети периода
            third = len(monthly_sales) // 3
            if third < 1:
                third = 1
            
            early_sales = monthly_sales['Qty'].iloc[:third].mean()
            recent_sales = monthly_sales['Qty'].iloc[-third:].mean()
            std_dev = monthly_sales['Qty'].std()
            mean_sales = monthly_sales['Qty'].mean()
            
            # Коэффициент вариации
            cv = std_dev / mean_sales if mean_sales > 0 else 0
            
            # Рост: последние продажи значительно выше начальных
            if recent_sales > early_sales * 1.2:
                stage = 'Рост'
            # Зрелость: стабильные продажи (низкая вариация)
            elif cv < 0.4:
                stage = 'Зрелость'
            # Спад: последние продажи ниже начальных
            elif recent_sales < early_sales * 0.8:
                stage = 'Спад'
            else:
                stage = 'Зрелость'
        else:
            # Для коротких периодов - анализ тренда
            if len(monthly_sales) >= 2:
                trend = monthly_sales['Qty'].iloc[-1] - monthly_sales['Qty'].iloc[0]
                if trend > 0:
                    stage = 'Рост'
                else:
                    stage = 'Зрелость'
            else:
                stage = 'Внедрение'
        
        lifecycle_data.append({
            'Art': art,
            'Describe': product_data['Describe'].iloc[0],
            'Total_Sales': int(total_sales),
            'Months_Active': months_active,
            'Stage': stage,
            'Avg_Monthly_Sales': round(avg_monthly_sales, 1)
        })
    
    return pd.DataFrame(lifecycle_data)

def generate_alerts(df, store, segment, recommendations):
    """Генерация алертов и уведомлений"""
    alerts = []
    
    if df.empty or df['Datasales'].isna().all():
        return alerts
    
    # Алерт 1: Товары с резким падением продаж
    max_date = df['Datasales'].max()
    min_date = df['Datasales'].min()
    
    # Проверка наличия достаточного периода данных
    if (max_date - min_date).days >= 60:
        recent_start = max_date - pd.Timedelta(days=30)
        previous_start = max_date - pd.Timedelta(days=60)
        previous_end = recent_start
        
        recent_data = df[(df['Datasales'] >= recent_start) & 
                        (df['Magazin'] == store) & 
                        (df['Segment'] == segment)]
        
        previous_data = df[(df['Datasales'] >= previous_start) & 
                          (df['Datasales'] < previous_end) &
                          (df['Magazin'] == store) & 
                          (df['Segment'] == segment)]
        
        if not recent_data.empty and not previous_data.empty:
            recent_sales = recent_data.groupby('Art')['Qty'].sum()
            previous_sales = previous_data.groupby('Art')['Qty'].sum()
            
            for art in recent_sales.index:
                if art in previous_sales.index and previous_sales[art] >= 5:  # Минимум 5 продаж
                    if recent_sales[art] == 0:
                        decline_pct = 100
                    else:
                        decline_pct = ((previous_sales[art] - recent_sales[art]) / previous_sales[art]) * 100
                    
                    if decline_pct >= 50:
                        product_name = df[df['Art'] == art]['Describe'].iloc[0]
                        alerts.append({
                            'type': 'warning',
                            'title': 'Падение продаж',
                            'message': f'Товар "{product_name}" ({art}): падение на {decline_pct:.0f}%',
                            'priority': 'high'
                        })
    
    # Алерт 2: Новые возможности
    if not recommendations.empty:
        top_opportunities = recommendations.head(5)
        for _, row in top_opportunities.iterrows():
            if row['Potential_Qty'] >= 10:  # Значимый потенциал
                alerts.append({
                    'type': 'success',
                    'title': 'Новая возможность',
                    'message': f'"{row["Describe"]}" ({row["Art"]}): потенциал {int(row["Potential_Qty"])} шт/мес',
                    'priority': 'medium'
                })
    
    # Алерт 3: Низкое покрытие ассортимента
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    
    if segment_unique > 0:
        coverage = (store_unique / segment_unique) * 100
        
        if coverage < 20:
            alerts.append({
                'type': 'error',
                'title': 'Критически низкое покрытие',
                'message': f'Покрытие ассортимента: {coverage:.1f}% (критический уровень)',
                'priority': 'high'
            })
        elif coverage < 40:
            alerts.append({
                'type': 'warning',
                'title': 'Низкое покрытие ассортимента',
                'message': f'Покрытие ассортимента: {coverage:.1f}% (требуется расширение)',
                'priority': 'medium'
            })
    
    # Алерт 4: Товары на стадии спада
    lifecycle_df = analyze_product_lifecycle(df, segment)
    store_declining = store_data['Art'].unique()
    
    if not lifecycle_df.empty:
        declining_products = lifecycle_df[
            (lifecycle_df['Stage'] == 'Спад') & 
            (lifecycle_df['Art'].isin(store_declining))
        ]
        
        if len(declining_products) > 0:
            alerts.append({
                'type': 'info',
                'title': 'Товары на стадии спада',
                'message': f'В ассортименте {len(declining_products)} товаров на стадии спада',
                'priority': 'low'
            })
    
    return alerts

def calculate_seasonality(df, segment):
    """Анализ сезонности для сегмента"""
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                       'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        return {
            'months': month_names,
            'sales': [0] * 12,
            'peak_month': 'Нет данных',
            'low_month': 'Нет данных',
            'seasonality_index': [100] * 12
        }
    
    # Группировка по месяцам
    monthly_sales = segment_data.groupby('Month')['Qty'].sum().reindex(range(1, 13), fill_value=0)
    
    if monthly_sales.sum() == 0:
        month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                       'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        return {
            'months': month_names,
            'sales': monthly_sales.values,
            'peak_month': 'Нет данных',
            'low_month': 'Нет данных',
            'seasonality_index': [100] * 12
        }
    
    # Расчет индекса сезонности (среднее = 100)
    avg_sales = monthly_sales.mean()
    seasonality_index = (monthly_sales / avg_sales * 100).values if avg_sales > 0 else [100] * 12
    
    peak_month = monthly_sales.idxmax()
    low_month = monthly_sales.idxmin()
    
    month_names = {1:'Янв', 2:'Фев', 3:'Мар', 4:'Апр', 5:'Май', 6:'Июн',
                   7:'Июл', 8:'Авг', 9:'Сен', 10:'Окт', 11:'Ноя', 12:'Дек'}
    
    month_labels = [month_names[i] for i in range(1, 13)]
    
    return {
        'months': month_labels,
        'sales': monthly_sales.values,
        'peak_month': month_names[peak_month],
        'low_month': month_names[low_month],
        'seasonality_index': seasonality_index
    }

def generate_recommendations_with_abc(df, store, segment, min_network_qty=10, max_store_qty=2):
    """Генерация рекомендаций с ABC анализом"""
    
    # Статистика по сети
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        return pd.DataFrame()
    
    # Агрегация по артикулам
    network_stats = segment_data.groupby('Art').agg({
        'Qty': 'sum',
        'Sum': 'sum',
        'Price': 'mean',
        'Describe': 'first',
        'Model': 'first',
        'Magazin': 'nunique'
    }).reset_index()
    
    network_stats.columns = ['Art', 'Total_Qty', 'Total_Sum', 'Avg_Price', 'Describe', 'Model', 'Store_Count']
    
    # Статистика по магазину
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)].copy()
    
    if store_data.empty:
        store_stats = pd.DataFrame(columns=['Art', 'Store_Qty'])
    else:
        store_stats = store_data.groupby('Art')['Qty'].sum().reset_index()
        store_stats.columns = ['Art', 'Store_Qty']
    
    # Объединение данных
    merged = network_stats.merge(store_stats, on='Art', how='left')
    merged['Store_Qty'] = merged['Store_Qty'].fillna(0)
    
    # Фильтрация по критериям
    filtered = merged[
        (merged['Total_Qty'] >= min_network_qty) &
        (merged['Store_Qty'] <= max_store_qty) &
        (merged['Store_Count'] > 0)  # Защита от деления на 0
    ].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    # ABC анализ
    abc_df = calculate_abc_analysis(df, segment)
    
    if not abc_df.empty and 'Art' in abc_df.columns and 'ABC' in abc_df.columns:
        filtered = filtered.merge(abc_df[['Art', 'ABC']], on='Art', how='left')
        filtered['ABC'] = filtered['ABC'].fillna('C')
    else:
        filtered['ABC'] = 'N/A'
    
    # Расчет потенциала (среднее по магазинам сети)
    filtered['Potential_Qty'] = (filtered['Total_Qty'] / filtered['Store_Count']).round(1)
    filtered['Potential_Sum'] = (filtered['Potential_Qty'] * filtered['Avg_Price']).round(2)
    
    # Приоритет по ABC
    abc_priority = {'A': 3, 'B': 2, 'C': 1, 'N/A': 0}
    filtered['Priority'] = filtered['ABC'].map(abc_priority).fillna(0)
    
    # Сортировка: приоритет ABC, затем потенциал
    filtered = filtered.sort_values(['Priority', 'Potential_Qty'], ascending=[False, False])
    
    return filtered

def create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df):
    """Создание таблицы статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    stats = []
    
    # Общая информация
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    coverage = (store_unique / segment_unique * 100) if segment_unique > 0 else 0
    
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Всего товаров в сегменте', 
                  'Значение': segment_unique})
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Товаров в магазине', 
                  'Значение': store_unique})
    stats.append({'Категория': 'Общая информация', 'Показатель': 'Покрытие ассортимента', 
                  'Значение': f"{coverage:.1f}%"})
    
    # Продажи
    total_segment_sales = segment_data['Qty'].sum()
    total_store_sales = store_data['Qty'].sum()
    segment_revenue = segment_data['Sum'].sum()
    store_revenue = store_data['Sum'].sum()
    
    stats.append({'Категория': 'Продажи', 'Показатель': 'Продажи сегмента (шт)', 
                  'Значение': f"{int(total_segment_sales):,}"})
    stats.append({'Категория': 'Продажи', 'Показатель': 'Продажи магазина (шт)', 
                  'Значение': f"{int(total_store_sales):,}"})
    stats.append({'Категория': 'Продажи', 'Показатель': 'Выручка сегмента (грн)', 
                  'Значение': f"{segment_revenue:,.0f}"})
    stats.append({'Категория': 'Продажи', 'Показатель': 'Выручка магазина (грн)', 
                  'Значение': f"{store_revenue:,.0f}"})
    
    # ABC анализ
    if not abc_df.empty:
        for category in ['A', 'B', 'C']:
            count = len(abc_df[abc_df['ABC'] == category])
            revenue = abc_df[abc_df['ABC'] == category]['Revenue'].sum()
            stats.append({'Категория': 'ABC Анализ', 
                         'Показатель': f'Категория {category} (товаров)', 
                         'Значение': count})
            stats.append({'Категория': 'ABC Анализ', 
                         'Показатель': f'Категория {category} (выручка)', 
                         'Значение': f"{revenue:,.0f} грн"})
    
    # Жизненный цикл
    if not lifecycle_df.empty:
        for stage in ['Внедрение', 'Рост', 'Зрелость', 'Спад']:
            count = len(lifecycle_df[lifecycle_df['Stage'] == stage])
            if count > 0:
                stats.append({'Категория': 'Жизненный цикл', 
                             'Показатель': stage, 
                             'Значение': count})
    
    # Рекомендации
    if not recommendations.empty:
        stats.append({'Категория': 'Рекомендации', 
                     'Показатель': 'Товаров рекомендовано', 
                     'Значение': len(recommendations)})
        stats.append({'Категория': 'Рекомендации', 
                     'Показатель': 'Потенциал продаж (шт/мес)', 
                     'Значение': f"{recommendations['Potential_Qty'].sum():.0f}"})
        stats.append({'Категория': 'Рекомендации', 
                     'Показатель': 'Потенциальная выручка (грн/мес)', 
                     'Значение': f"{recommendations['Potential_Sum'].sum():,.0f}"})
    
    return pd.DataFrame(stats)

def create_excel_report(df, store, segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts):
    """Создание Excel отчета"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Рекомендации
        if not recommendations.empty:
            rec_export = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 
                                         'Store_Qty', 'Potential_Qty', 'Potential_Sum', 
                                         'Store_Count', 'ABC']].copy()
            rec_export.columns = ['Артикул', 'Описание', 'Модель', 'Цена', 'Продажи сети', 
                                 'Продажи магазина', 'Потенциал (шт)', 'Потенциал (грн)', 
                                 'Магазинов', 'ABC']
            rec_export.to_excel(writer, sheet_name='Рекомендации', index=False)
        
        # Лист 2: Статистика
        stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)
        stats_table.to_excel(writer, sheet_name='Статистика', index=False)
        
        # Лист 3: ABC анализ
        if not abc_df.empty:
            abc_export = abc_df.copy()
            abc_export.columns = ['Артикул', 'Выручка', 'ABC', 'Кумулятивный %']
            abc_export.to_excel(writer, sheet_name='ABC Анализ', index=False)
        
        # Лист 4: Жизненный цикл
        if not lifecycle_df.empty:
            lifecycle_export = lifecycle_df.copy()
            lifecycle_export.columns = ['Артикул', 'Описание', 'Всего продаж', 
                                       'Месяцев активности', 'Стадия', 'Средние продажи/мес']
            lifecycle_export.to_excel(writer, sheet_name='Жизненный цикл', index=False)
        
        # Лист 5: Сезонность
        season_df = pd.DataFrame({
            'Месяц': seasonality_data['months'],
            'Продажи': seasonality_data['sales'],
            'Индекс сезонности': seasonality_data['seasonality_index']
        })
        season_df.to_excel(writer, sheet_name='Сезонность', index=False)
        
        # Лист 6: Алерты
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
    
    # Сортировка по приоритету
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_alerts = sorted(alerts, key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
    
    for alert in sorted_alerts:
        alert_type = alert.get('type', 'info')
        title = alert.get('title', 'Уведомление')
        message = alert.get('message', '')
        
        if alert_type == 'error':
            st.error(f"**{title}**: {message}")
        elif alert_type == 'warning':
            st.warning(f"**{title}**: {message}")
        elif alert_type == 'success':
            st.success(f"**{title}**: {message}")
        else:
            st.info(f"**{title}**: {message}")

def display_results(df, store, segment, recommendations, seasonality_data, lifecycle_df, alerts, abc_df):
    """Отображение результатов"""
    
    # Алерты
    if alerts:
        display_alerts(alerts)
        st.divider()
    
    # Статистика в метриках
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    coverage = (store_unique / segment_unique * 100) if segment_unique > 0 else 0
    
    avg_sales_network = segment_data.groupby('Art')['Qty'].sum().mean() if not segment_data.empty else 0
    avg_price = segment_data['Price'].mean() if not segment_data.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Товаров в сегменте", segment_unique)
    with col2:
        st.metric("Товаров в магазине", store_unique)
    with col3:
        st.metric("Покрытие ассортимента", f"{coverage:.1f}%")
    with col4:
        st.metric("Средняя цена", f"{avg_price:.0f} грн")
    
    st.divider()
    
    # Рекомендации
    st.subheader("🎯 Рекомендации товаров для добавления")
    
    if not recommendations.empty:
        # Фильтры для рекомендаций
        col1, col2 = st.columns(2)
        with col1:
            abc_filter = st.multiselect(
                "Фильтр по ABC",
                options=['A', 'B', 'C', 'N/A'],
                default=['A', 'B', 'C']
            )
        with col2:
            min_potential = st.slider(
                "Минимальный потенциал (шт)",
                min_value=0,
                max_value=int(recommendations['Potential_Qty'].max()),
                value=0
            )
        
        # Применение фильтров
        filtered_rec = recommendations[
            (recommendations['ABC'].isin(abc_filter)) &
            (recommendations['Potential_Qty'] >= min_potential)
        ].copy()
        
        if not filtered_rec.empty:
            display_df = filtered_rec[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty', 
                                      'Store_Qty', 'Potential_Qty', 'Store_Count', 'ABC']].copy()
            display_df.columns = ['Артикул', 'Описание', 'Модель', 'Цена (грн)', 'Продажи сети', 
                                 'Продажи магазина', 'Потенциал (шт/мес)', 'Магазинов', 'ABC']
            
            # Форматирование
            display_df['Цена (грн)'] = display_df['Цена (грн)'].apply(lambda x: f"{x:.2f}")
            display_df['Продажи сети'] = display_df['Продажи сети'].apply(lambda x: f"{int(x):,}")
            display_df['Продажи магазина'] = display_df['Продажи магазина'].apply(lambda x: f"{int(x):,}")
            display_df['Потенциал (шт/мес)'] = display_df['Потенциал (шт/мес)'].apply(lambda x: f"{x:.1f}")
            
            # Цветовое выделение ABC
            def color_abc(val):
                colors = {
                    'A': 'background-color: #90EE90',
                    'B': 'background-color: #FFE4B5',
                    'C': 'background-color: #FFB6C1',
                    'N/A': 'background-color: #E0E0E0'
                }
                return colors.get(val, '')
            
            styled_df = display_df.style.applymap(color_abc, subset=['ABC'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Метрики по рекомендациям
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Рекомендовано товаров", len(filtered_rec))
            with col2:
                st.metric("Потенциал (шт/мес)", f"{filtered_rec['Potential_Qty'].sum():.0f}")
            with col3:
                st.metric("Потенциал (грн/мес)", f"{filtered_rec['Potential_Sum'].sum():,.0f}")
        else:
            st.info("Нет товаров, соответствующих выбранным фильтрам")
    else:
        st.info("Рекомендации не найдены. Попробуйте изменить параметры.")
    
    st.divider()
    
    # ABC анализ
    st.subheader("📊 ABC Анализ сегмента")
    
    if not abc_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # График ABC
            abc_counts = abc_df['ABC'].value_counts().reindex(['A', 'B', 'C'], fill_value=0)
            fig_abc = px.pie(
                values=abc_counts.values,
                names=abc_counts.index,
                title="Распределение товаров по категориям ABC",
                color=abc_counts.index,
                color_discrete_map={'A': '#90EE90', 'B': '#FFE4B5', 'C': '#FFB6C1'}
            )
            st.plotly_chart(fig_abc, use_container_width=True)
        
        with col2:
            st.markdown("**Категории ABC:**")
            for category in ['A', 'B', 'C']:
                count = len(abc_df[abc_df['ABC'] == category])
                revenue = abc_df[abc_df['ABC'] == category]['Revenue'].sum()
                pct = (revenue / abc_df['Revenue'].sum() * 100) if abc_df['Revenue'].sum() > 0 else 0
                st.metric(f"Категория {category}", f"{count} товаров", f"{pct:.1f}% выручки")
    else:
        st.info("Недостаточно данных для ABC анализа")
    
    st.divider()
    
    # Жизненный цикл
    st.subheader("🔄 Анализ жизненного цикла товаров")
    
    if not lifecycle_df.empty:
        # Метрики по стадиям
        stage_summary = lifecycle_df['Stage'].value_counts()
        col1, col2, col3, col4 = st.columns(4)
        
        stages = ['Внедрение', 'Рост', 'Зрелость', 'Спад']
        icons = ['🚀', '📈', '⚖️', '📉']
        cols = [col1, col2, col3, col4]
        
        for i, (stage, icon) in enumerate(zip(stages, icons)):
            with cols[i]:
                count = stage_summary.get(stage, 0)
                st.metric(f"{icon} {stage}", count)
        
        # График распределения
        fig_lifecycle = px.pie(
            values=stage_summary.values,
            names=stage_summary.index,
            title="Распределение товаров по стадиям жизненного цикла"
        )
        st.plotly_chart(fig_lifecycle, use_container_width=True)
        
        # Таблица товаров
        with st.expander("📋 Подробная информация по товарам"):
            lifecycle_display = lifecycle_df[['Art', 'Describe', 'Stage', 'Total_Sales', 
                                             'Months_Active', 'Avg_Monthly_Sales']].copy()
            lifecycle_display.columns = ['Артикул', 'Описание', 'Стадия', 'Всего продаж', 
                                        'Месяцев активности', 'Средние продажи/мес']
            st.dataframe(lifecycle_display, use_container_width=True)
    else:
        st.info("Недостаточно данных для анализа жизненного цикла")
    
    st.divider()
    
    # Сезонность
    st.subheader("📅 Анализ сезонности продаж")
    
    if seasonality_data['peak_month'] != 'Нет данных':
        # График продаж по месяцам
        fig_season = go.Figure()
        
        fig_season.add_trace(go.Scatter(
            x=seasonality_data['months'],
            y=seasonality_data['sales'],
            mode='lines+markers',
            name='Продажи',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig_season.update_layout(
            title="Сезонность продаж в сегменте",
            xaxis_title="Месяц",
            yaxis_title="Количество продаж",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Индекс сезонности
        fig_index = go.Figure()
        
        fig_index.add_trace(go.Bar(
            x=seasonality_data['months'],
            y=seasonality_data['seasonality_index'],
            marker_color=['#90EE90' if x >= 100 else '#FFB6C1' for x in seasonality_data['seasonality_index']],
            text=[f"{x:.0f}" for x in seasonality_data['seasonality_index']],
            textposition='outside'
        ))
        
        fig_index.add_hline(y=100, line_dash="dash", line_color="gray", 
                           annotation_text="Среднее значение")
        
        fig_index.update_layout(
            title="Индекс сезонности (среднее = 100)",
            xaxis_title="Месяц",
            yaxis_title="Индекс",
            showlegend=False
        )
        
        st.plotly_chart(fig_index, use_container_width=True)
        
        # Информация о пиках
        col1, col2 = st.columns(2)
        with col1:
            peak_value = seasonality_data['sales'][seasonality_data['months'].index(seasonality_data['peak_month'])]
            st.success(f"📈 **Пиковый месяц:** {seasonality_data['peak_month']} ({int(peak_value)} шт)")
        with col2:
            low_value = seasonality_data['sales'][seasonality_data['months'].index(seasonality_data['low_month'])]
            st.info(f"📉 **Низкий месяц:** {seasonality_data['low_month']} ({int(low_value)} шт)")
    else:
        st.info("Недостаточно данных для анализа сезонности")
    
    st.divider()
    
    # Детальная статистика
    st.subheader("📈 Детальная статистика")
    
    stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)
    
    if not stats_table.empty:
        categories = stats_table['Категория'].unique().tolist()
        
        if len(categories) > 0:
            tabs = st.tabs(categories)
            
            for i, category in enumerate(categories):
                with tabs[i]:
                    category_data = stats_table[stats_table['Категория'] == category]
                    category_display = category_data[['Показатель', 'Значение']].copy()
                    
                    # Красивое отображение таблицы
                    st.dataframe(
                        category_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Показатель": st.column_config.TextColumn("Показатель", width="medium"),
                            "Значение": st.column_config.TextColumn("Значение", width="medium")
                        }
                    )
        else:
            st.info("Нет данных для отображения статистики")

def main():
    # Заголовок
    st.title("🛍️ Рекомендательная система товаров, которая предлагает магазину перечень товаров, которые хорошо продаются в сети,но еще не представлены в данном магазине")
    st.markdown("""
    Система анализа и рекомендаций товаров с использованием:
    - **ABC анализ** - классификация по выручке
    - **Анализ жизненного цикла** - определение стадии товара
    - **Анализ сезонности** - выявление сезонных паттернов
    - **Интеллектуальные алерты** - уведомления о важных событиях
    """)
    
    # Выбор источника данных
    st.subheader("📥 Источник данных")
    data_source = st.radio(
        "Выберите источник данных:",
        options=["Локальный файл", "Google Sheets"],
        horizontal=True
    )

    df = None

    if data_source == "Локальный файл":
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "📁 Загрузите Excel файл с данными о продажах",
            type=['xlsx', 'xls'],
            help="Файл должен содержать колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
        )

        if uploaded_file is None:
            st.info("👆 Загрузите Excel файл для начала работы")
            with st.expander("ℹ️ Требования к формату данных"):
                st.markdown("""
                **Обязательные колонки:**
                - `Magazin` - название магазина
                - `Datasales` - дата продажи
                - `Art` - артикул товара
                - `Describe` - описание товара
                - `Model` - модель товара
                - `Segment` - сегмент товара
                - `Price` - цена
                - `Qty` - количество
                - `Sum` - сумма продажи

                **Форматы дат:** DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD
                """)
            return

        # Загрузка данных из файла
        with st.spinner("⏳ Загрузка и обработка данных..."):
            df = load_and_process_data(uploaded_file)

    else:  # Google Sheets
        # Поле для ввода URL
        sheet_url = st.text_input(
            "URL Google Sheets:",
            value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit?gid=64159818#gid=64159818",
            help="Таблица должна быть открыта для просмотра (публичный доступ)"
        )

        if not sheet_url:
            st.info("👆 Введите URL таблицы Google Sheets для начала работы")
            with st.expander("ℹ️ Как получить публичный доступ к таблице?"):
                st.markdown("""
                **Инструкция:**
                1. Откройте вашу таблицу в Google Sheets
                2. Нажмите кнопку "Настройки доступа" (справа вверху)
                3. Выберите "Все, у кого есть ссылка"
                4. Установите права "Читатель"
                5. Скопируйте ссылку на таблицу

                **Обязательные колонки:**
                - `Magazin`, `Datasales`, `Art`, `Describe`, `Model`, `Segment`, `Price`, `Qty`, `Sum`

                **Форматы дат:** DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD
                """)
            return

        # Загрузка данных из Google Sheets
        with st.spinner("⏳ Загрузка и обработка данных из Google Sheets..."):
            df = load_data_from_google_sheets(sheet_url)
    
    if df is None:
        return
    
    st.success(f"✅ Данные успешно загружены: {len(df):,} записей")
    
    # Информация о данных
    with st.expander("📊 Информация о загруженных данных"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Записей", f"{len(df):,}")
        with col2:
            st.metric("Магазинов", df['Magazin'].nunique())
        with col3:
            st.metric("Товаров", df['Art'].nunique())
        with col4:
            st.metric("Сегментов", df['Segment'].nunique())
        
        date_range = f"{df['Datasales'].min().strftime('%d.%m.%Y')} - {df['Datasales'].max().strftime('%d.%m.%Y')}"
        st.info(f"📅 Период данных: {date_range}")
    
    # Боковая панель с параметрами
    with st.sidebar:
        st.header("⚙️ Параметры анализа")
        
        # Выбор магазина и сегмента
        stores = sorted(df['Magazin'].unique())
        segments = sorted(df['Segment'].unique())
        
        selected_store = st.selectbox(
            "🏪 Выберите магазин:",
            stores,
            help="Магазин для анализа"
        )
        
        selected_segment = st.selectbox(
            "📊 Выберите сегмент:",
            segments,
            help="Сегмент товаров"
        )
        
        st.divider()
        
        # Критерии рекомендаций
        st.subheader("🎯 Критерии рекомендаций")
        
        min_network_qty = st.number_input(
            "Минимальные продажи в сети:",
            min_value=1,
            max_value=1000,
            value=10,
            step=5,
            help="Минимальное количество продаж товара в сети для рекомендации"
        )
        
        max_store_qty = st.number_input(
            "Максимальные продажи в магазине:",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            help="Максимальное количество продаж в магазине (0 = товар отсутствует)"
        )
        
        st.divider()
        
        # Кнопка анализа
        analyze_btn = st.button(
            "🎯 Запустить анализ",
            type="primary",
            use_container_width=True
        )
    
    # Анализ
    if analyze_btn:
        with st.spinner("🔍 Выполняется анализ данных..."):
            try:
                # Генерация рекомендаций
                recommendations = generate_recommendations_with_abc(
                    df, selected_store, selected_segment, min_network_qty, max_store_qty
                )
                
                # ABC анализ
                abc_df = calculate_abc_analysis(df, selected_segment)
                
                # Сезонность
                seasonality_data = calculate_seasonality(df, selected_segment)
                
                # Жизненный цикл
                lifecycle_df = analyze_product_lifecycle(df, selected_segment)
                
                # Алерты
                alerts = generate_alerts(df, selected_store, selected_segment, recommendations)
                
                # Отображение результатов
                st.success("✅ Анализ завершен успешно!")
                st.divider()
                
                display_results(
                    df, selected_store, selected_segment, 
                    recommendations, seasonality_data, 
                    lifecycle_df, alerts, abc_df
                )
                
                # Экспорт отчета
                st.divider()
                st.subheader("📥 Экспорт отчета")
                
                excel_report = create_excel_report(
                    df, selected_store, selected_segment,
                    recommendations, abc_df, seasonality_data,
                    lifecycle_df, alerts
                )
                
                filename = f"analysis_report_{selected_store}_{selected_segment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                
                st.download_button(
                    label="📊 Скачать полный отчет Excel",
                    data=excel_report.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Ошибка при анализе: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()


