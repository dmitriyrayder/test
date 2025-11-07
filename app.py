import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Конфігурація сторінки
st.set_page_config(page_title="Рекомендаційна система товарів, яка пропонує магазину перелік товарів, які добре продаються в мережі, але ще не представлені в даному магазині", page_icon="🛍️", layout="wide")

@st.cache_data
def load_and_process_data(uploaded_file):
    """Завантаження та попередня обробка даних"""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Відсутні колонки: {missing_cols}")
            return None

        # Обробка дат
        # Зберігаємо оригінальний стовпець для повторних спроб парсингу
        datasales_original = df['Datasales'].copy()
        date_formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%y', '%d/%m/%y']
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')

        if df['Datasales'].isna().all():
            for fmt in date_formats:
                try:
                    # Застосовуємо формат до ОРИГІНАЛЬНИХ рядкових значень
                    df['Datasales'] = pd.to_datetime(datasales_original, format=fmt, errors='coerce')
                    if not df['Datasales'].isna().all():
                        break
                except:
                    continue

        # Очищення даних
        df = df.dropna(subset=['Art', 'Magazin', 'Segment', 'Datasales'])

        # Перевірка, чи залишились дані після очищення
        if df.empty:
            st.error("Після обробки не залишилось валідних даних. Перевірте формат даних у файлі.")
            return None

        # Конвертація числових колонок
        for col in ['Qty', 'Price', 'Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Видалення дублікатів та некоректних даних
        df = df[df['Qty'] > 0]  # Тільки позитивні продажі
        df = df[df['Price'] > 0]  # Тільки позитивні ціни
        df = df.drop_duplicates()

        # Перевірка після фільтрації
        if df.empty:
            st.error("Після фільтрації не залишилось даних. Перевірте, що є записи з позитивними Qty та Price.")
            return None

        # Додавання часових ознак
        df['Month'] = df['Datasales'].dt.month
        df['Year'] = df['Datasales'].dt.year
        df['Week'] = df['Datasales'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"Помилка завантаження даних: {str(e)}")
        return None

@st.cache_data
def load_data_from_google_sheets(sheet_url):
    """Завантаження даних з публічної таблиці Google Sheets"""
    try:
        # Витягування ID таблиці з URL
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not match:
            st.error("Невірний формат URL Google Sheets")
            return None

        sheet_id = match.group(1)

        # Формування URL для експорту в CSV форматі
        export_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'

        # Завантаження даних
        df = pd.read_csv(export_url)

        # Перевірка обов'язкових колонок
        required_cols = ['Magazin', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Відсутні колонки: {missing_cols}")
            return None

        # Обробка дат
        # Зберігаємо оригінальний стовпець для повторних спроб парсингу
        datasales_original = df['Datasales'].copy()
        date_formats = ['%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%y', '%d/%m/%y']
        df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')

        if df['Datasales'].isna().all():
            for fmt in date_formats:
                try:
                    # Застосовуємо формат до ОРИГІНАЛЬНИХ рядкових значень
                    df['Datasales'] = pd.to_datetime(datasales_original, format=fmt, errors='coerce')
                    if not df['Datasales'].isna().all():
                        break
                except:
                    continue

        # Очищення даних
        df = df.dropna(subset=['Art', 'Magazin', 'Segment', 'Datasales'])

        # Перевірка, чи залишились дані після очищення
        if df.empty:
            st.error("Після обробки не залишилось валідних даних. Перевірте формат даних у таблиці.")
            return None

        # Конвертація числових колонок
        for col in ['Qty', 'Price', 'Sum']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Видалення дублікатів та некоректних даних
        df = df[df['Qty'] > 0]  # Тільки позитивні продажі
        df = df[df['Price'] > 0]  # Тільки позитивні ціни
        df = df.drop_duplicates()

        # Перевірка після фільтрації
        if df.empty:
            st.error("Після фільтрації не залишилось даних. Перевірте, що в таблиці є записи з позитивними Qty та Price.")
            return None

        # Додавання часових ознак
        df['Month'] = df['Datasales'].dt.month
        df['Year'] = df['Datasales'].dt.year
        df['Week'] = df['Datasales'].dt.isocalendar().week

        return df

    except Exception as e:
        st.error(f"Помилка завантаження даних з Google Sheets: {str(e)}")
        return None

def calculate_abc_analysis(df, segment):
    """ABC аналіз для сегмента за методом Парето"""
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        return pd.DataFrame(columns=['Art', 'Revenue', 'ABC', 'Cumulative_Pct'])

    # Групування по артикулу
    product_revenue = segment_data.groupby('Art')['Sum'].sum().sort_values(ascending=False)

    if product_revenue.empty or product_revenue.sum() == 0:
        return pd.DataFrame(columns=['Art', 'Revenue', 'ABC', 'Cumulative_Pct'])

    # Розрахунок кумулятивного відсотка
    total_revenue = product_revenue.sum()
    cumulative_revenue = product_revenue.cumsum()
    cumulative_percentage = (cumulative_revenue / total_revenue) * 100

    # Присвоєння категорій ABC
    # A: 0-80% виручки, B: 80-95%, C: 95-100%
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
    """Аналіз життєвого циклу товарів"""
    segment_data = df[df['Segment'] == segment].copy()
    
    if segment_data.empty:
        return pd.DataFrame(columns=['Art', 'Describe', 'Total_Sales', 'Months_Active', 'Stage', 'Avg_Monthly_Sales'])
    
    lifecycle_data = []
    
    for art in segment_data['Art'].unique():
        product_data = segment_data[segment_data['Art'] == art].copy()

        # Групування по місяцях
        monthly_sales = product_data.groupby(['Year', 'Month'])['Qty'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values(['Year', 'Month'])

        if len(monthly_sales) == 0:
            continue

        total_sales = monthly_sales['Qty'].sum()
        months_active = len(monthly_sales[monthly_sales['Qty'] > 0])
        avg_monthly_sales = total_sales / months_active if months_active > 0 else 0

        # Визначення стадії життєвого циклу
        if months_active <= 2:
            stage = 'Впровадження'
        elif len(monthly_sales) >= 4:
            # Беремо першу та останню третини періоду
            third = len(monthly_sales) // 3
            if third < 1:
                third = 1

            early_sales = monthly_sales['Qty'].iloc[:third].mean()
            recent_sales = monthly_sales['Qty'].iloc[-third:].mean()
            std_dev = monthly_sales['Qty'].std()
            mean_sales = monthly_sales['Qty'].mean()

            # Коефіцієнт варіації
            cv = std_dev / mean_sales if mean_sales > 0 else 0

            # Зростання: останні продажі значно вищі початкових
            if recent_sales > early_sales * 1.2:
                stage = 'Зростання'
            # Зрілість: стабільні продажі (низька варіація)
            elif cv < 0.4:
                stage = 'Зрілість'
            # Спад: останні продажі нижчі початкових
            elif recent_sales < early_sales * 0.8:
                stage = 'Спад'
            else:
                stage = 'Зрілість'
        else:
            # Для коротких періодів - аналіз тренду
            if len(monthly_sales) >= 2:
                trend = monthly_sales['Qty'].iloc[-1] - monthly_sales['Qty'].iloc[0]
                if trend > 0:
                    stage = 'Зростання'
                else:
                    stage = 'Зрілість'
            else:
                stage = 'Впровадження'
        
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
    """Генерація алертів та сповіщень"""
    alerts = []

    if df.empty or df['Datasales'].isna().all():
        return alerts

    # Алерт 1: Товари з різким падінням продажів
    max_date = df['Datasales'].max()
    min_date = df['Datasales'].min()

    # Перевірка наявності достатнього періоду даних
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
                if art in previous_sales.index and previous_sales[art] >= 5:  # Мінімум 5 продажів
                    if recent_sales[art] == 0:
                        decline_pct = 100
                    else:
                        decline_pct = ((previous_sales[art] - recent_sales[art]) / previous_sales[art]) * 100

                    if decline_pct >= 50:
                        product_name = df[df['Art'] == art]['Describe'].iloc[0]
                        alerts.append({
                            'type': 'warning',
                            'title': 'Падіння продажів',
                            'message': f'Товар "{product_name}" ({art}): падіння на {decline_pct:.0f}%',
                            'priority': 'high'
                        })

    # Алерт 2: Нові можливості
    if not recommendations.empty:
        top_opportunities = recommendations.head(5)
        for _, row in top_opportunities.iterrows():
            if row['Potential_Qty'] >= 10:  # Значний потенціал
                alerts.append({
                    'type': 'success',
                    'title': 'Нова можливість',
                    'message': f'"{row["Describe"]}" ({row["Art"]}): потенціал {int(row["Potential_Qty"])} шт/міс',
                    'priority': 'medium'
                })

    # Алерт 3: Низьке покриття асортименту
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]
    
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    
    if segment_unique > 0:
        coverage = (store_unique / segment_unique) * 100

        if coverage < 20:
            alerts.append({
                'type': 'error',
                'title': 'Критично низьке покриття',
                'message': f'Покриття асортименту: {coverage:.1f}% (критичний рівень)',
                'priority': 'high'
            })
        elif coverage < 40:
            alerts.append({
                'type': 'warning',
                'title': 'Низьке покриття асортименту',
                'message': f'Покриття асортименту: {coverage:.1f}% (потрібне розширення)',
                'priority': 'medium'
            })

    # Алерт 4: Товари на стадії спаду
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
                'title': 'Товари на стадії спаду',
                'message': f'В асортименті {len(declining_products)} товарів на стадії спаду',
                'priority': 'low'
            })
    
    return alerts

def calculate_seasonality(df, segment):
    """Аналіз сезонності для сегмента"""
    segment_data = df[df['Segment'] == segment].copy()

    if segment_data.empty:
        month_names = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер',
                       'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
        return {
            'months': month_names,
            'sales': [0] * 12,
            'peak_month': 'Немає даних',
            'low_month': 'Немає даних',
            'seasonality_index': [100] * 12
        }

    # Групування по місяцях
    monthly_sales = segment_data.groupby('Month')['Qty'].sum().reindex(range(1, 13), fill_value=0)

    if monthly_sales.sum() == 0:
        month_names = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер',
                       'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
        return {
            'months': month_names,
            'sales': monthly_sales.values,
            'peak_month': 'Немає даних',
            'low_month': 'Немає даних',
            'seasonality_index': [100] * 12
        }

    # Розрахунок індексу сезонності (середнє = 100)
    avg_sales = monthly_sales.mean()
    seasonality_index = (monthly_sales / avg_sales * 100).values if avg_sales > 0 else [100] * 12

    peak_month = monthly_sales.idxmax()
    low_month = monthly_sales.idxmin()

    month_names = {1:'Січ', 2:'Лют', 3:'Бер', 4:'Кві', 5:'Тра', 6:'Чер',
                   7:'Лип', 8:'Сер', 9:'Вер', 10:'Жов', 11:'Лис', 12:'Гру'}
    
    month_labels = [month_names[i] for i in range(1, 13)]
    
    return {
        'months': month_labels,
        'sales': monthly_sales.values,
        'peak_month': month_names[peak_month],
        'low_month': month_names[low_month],
        'seasonality_index': seasonality_index
    }

def generate_recommendations_with_abc(df, store, segment, min_network_qty=10, max_store_qty=2):
    """Генерація рекомендацій з ABC аналізом"""

    # Статистика по мережі
    segment_data = df[df['Segment'] == segment].copy()

    if segment_data.empty:
        return pd.DataFrame()

    # Агрегація по артикулам
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

    # Об'єднання даних
    merged = network_stats.merge(store_stats, on='Art', how='left')
    merged['Store_Qty'] = merged['Store_Qty'].fillna(0)

    # Фільтрація за критеріями
    filtered = merged[
        (merged['Total_Qty'] >= min_network_qty) &
        (merged['Store_Qty'] <= max_store_qty) &
        (merged['Store_Count'] > 0)  # Захист від ділення на 0
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    # ABC аналіз
    abc_df = calculate_abc_analysis(df, segment)

    if not abc_df.empty and 'Art' in abc_df.columns and 'ABC' in abc_df.columns:
        filtered = filtered.merge(abc_df[['Art', 'ABC']], on='Art', how='left')
        filtered['ABC'] = filtered['ABC'].fillna('C')
    else:
        filtered['ABC'] = 'N/A'

    # Розрахунок потенціалу (середнє по магазинах мережі)
    filtered['Potential_Qty'] = (filtered['Total_Qty'] / filtered['Store_Count']).round(1)
    filtered['Potential_Sum'] = (filtered['Potential_Qty'] * filtered['Avg_Price']).round(2)

    # Пріоритет по ABC
    abc_priority = {'A': 3, 'B': 2, 'C': 1, 'N/A': 0}
    filtered['Priority'] = filtered['ABC'].map(abc_priority).fillna(0)

    # Сортування: пріоритет ABC, потім потенціал
    filtered = filtered.sort_values(['Priority', 'Potential_Qty'], ascending=[False, False])
    
    return filtered

def create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df):
    """Створення таблиці статистики"""
    segment_data = df[df['Segment'] == segment]
    store_data = df[(df['Magazin'] == store) & (df['Segment'] == segment)]

    stats = []

    # Загальна інформація
    segment_unique = segment_data['Art'].nunique()
    store_unique = store_data['Art'].nunique()
    coverage = (store_unique / segment_unique * 100) if segment_unique > 0 else 0

    stats.append({'Категорія': 'Загальна інформація', 'Показник': 'Всього товарів в сегменті',
                  'Значення': segment_unique})
    stats.append({'Категорія': 'Загальна інформація', 'Показник': 'Товарів у магазині',
                  'Значення': store_unique})
    stats.append({'Категорія': 'Загальна інформація', 'Показник': 'Покриття асортименту',
                  'Значення': f"{coverage:.1f}%"})

    # Продажі
    total_segment_sales = segment_data['Qty'].sum()
    total_store_sales = store_data['Qty'].sum()
    segment_revenue = segment_data['Sum'].sum()
    store_revenue = store_data['Sum'].sum()

    stats.append({'Категорія': 'Продажі', 'Показник': 'Продажі сегмента (шт)',
                  'Значення': f"{int(total_segment_sales):,}"})
    stats.append({'Категорія': 'Продажі', 'Показник': 'Продажі магазину (шт)',
                  'Значення': f"{int(total_store_sales):,}"})
    stats.append({'Категорія': 'Продажі', 'Показник': 'Виручка сегмента (грн)',
                  'Значення': f"{segment_revenue:,.0f}"})
    stats.append({'Категорія': 'Продажі', 'Показник': 'Виручка магазину (грн)',
                  'Значення': f"{store_revenue:,.0f}"})

    # ABC аналіз
    if not abc_df.empty:
        for category in ['A', 'B', 'C']:
            count = len(abc_df[abc_df['ABC'] == category])
            revenue = abc_df[abc_df['ABC'] == category]['Revenue'].sum()
            stats.append({'Категорія': 'ABC Аналіз',
                         'Показник': f'Категорія {category} (товарів)',
                         'Значення': count})
            stats.append({'Категорія': 'ABC Аналіз',
                         'Показник': f'Категорія {category} (виручка)',
                         'Значення': f"{revenue:,.0f} грн"})

    # Життєвий цикл
    if not lifecycle_df.empty:
        for stage in ['Впровадження', 'Зростання', 'Зрілість', 'Спад']:
            count = len(lifecycle_df[lifecycle_df['Stage'] == stage])
            if count > 0:
                stats.append({'Категорія': 'Життєвий цикл',
                             'Показник': stage,
                             'Значення': count})

    # Рекомендації
    if not recommendations.empty:
        stats.append({'Категорія': 'Рекомендації',
                     'Показник': 'Товарів рекомендовано',
                     'Значення': len(recommendations)})
        stats.append({'Категорія': 'Рекомендації',
                     'Показник': 'Потенціал продажів (шт/міс)',
                     'Значення': f"{recommendations['Potential_Qty'].sum():.0f}"})
        stats.append({'Категорія': 'Рекомендації',
                     'Показник': 'Потенційна виручка (грн/міс)',
                     'Значення': f"{recommendations['Potential_Sum'].sum():,.0f}"})
    
    return pd.DataFrame(stats)

def create_excel_report(df, store, segment, recommendations, abc_df, seasonality_data, lifecycle_df, alerts):
    """Створення Excel звіту"""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Лист 1: Рекомендації
        if not recommendations.empty:
            rec_export = recommendations[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty',
                                         'Store_Qty', 'Potential_Qty', 'Potential_Sum',
                                         'Store_Count', 'ABC']].copy()
            rec_export.columns = ['Артикул', 'Опис', 'Модель', 'Ціна', 'Продажі мережі',
                                 'Продажі магазину', 'Потенціал (шт)', 'Потенціал (грн)',
                                 'Магазинів', 'ABC']
            rec_export.to_excel(writer, sheet_name='Рекомендації', index=False)

        # Лист 2: Статистика
        stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)
        stats_table.to_excel(writer, sheet_name='Статистика', index=False)

        # Лист 3: ABC аналіз
        if not abc_df.empty:
            abc_export = abc_df.copy()
            abc_export.columns = ['Артикул', 'Виручка', 'ABC', 'Кумулятивний %']
            abc_export.to_excel(writer, sheet_name='ABC Аналіз', index=False)

        # Лист 4: Життєвий цикл
        if not lifecycle_df.empty:
            lifecycle_export = lifecycle_df.copy()
            lifecycle_export.columns = ['Артикул', 'Опис', 'Всього продажів',
                                       'Місяців активності', 'Стадія', 'Середні продажі/міс']
            lifecycle_export.to_excel(writer, sheet_name='Життєвий цикл', index=False)

        # Лист 5: Сезонність
        season_df = pd.DataFrame({
            'Місяць': seasonality_data['months'],
            'Продажі': seasonality_data['sales'],
            'Індекс сезонності': seasonality_data['seasonality_index']
        })
        season_df.to_excel(writer, sheet_name='Сезонність', index=False)

        # Лист 6: Алерти
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df.to_excel(writer, sheet_name='Алерти', index=False)
    
    output.seek(0)
    return output

def display_alerts(alerts):
    """Відображення алертів"""
    if not alerts:
        return

    st.subheader("🚨 Алерти та сповіщення")

    # Сортування за пріоритетом
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_alerts = sorted(alerts, key=lambda x: priority_order.get(x.get('priority', 'low'), 2))

    for alert in sorted_alerts:
        alert_type = alert.get('type', 'info')
        title = alert.get('title', 'Сповіщення')
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
    """Відображення результатів"""

    # Алерти
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
        st.metric("Товарів в сегменті", segment_unique)
    with col2:
        st.metric("Товарів у магазині", store_unique)
    with col3:
        st.metric("Покриття асортименту", f"{coverage:.1f}%")
    with col4:
        st.metric("Середня ціна", f"{avg_price:.0f} грн")

    st.divider()

    # Рекомендації
    st.subheader("🎯 Рекомендації товарів для додавання")
    
    if not recommendations.empty:
        # Фільтри для рекомендацій
        col1, col2 = st.columns(2)
        with col1:
            abc_filter = st.multiselect(
                "Фільтр за ABC",
                options=['A', 'B', 'C', 'N/A'],
                default=['A', 'B', 'C']
            )
        with col2:
            min_potential = st.slider(
                "Мінімальний потенціал (шт)",
                min_value=0,
                max_value=int(recommendations['Potential_Qty'].max()),
                value=0
            )

        # Застосування фільтрів
        filtered_rec = recommendations[
            (recommendations['ABC'].isin(abc_filter)) &
            (recommendations['Potential_Qty'] >= min_potential)
        ].copy()

        if not filtered_rec.empty:
            display_df = filtered_rec[['Art', 'Describe', 'Model', 'Avg_Price', 'Total_Qty',
                                      'Store_Qty', 'Potential_Qty', 'Store_Count', 'ABC']].copy()
            display_df.columns = ['Артикул', 'Опис', 'Модель', 'Ціна (грн)', 'Продажі мережі',
                                 'Продажі магазину', 'Потенціал (шт/міс)', 'Магазинів', 'ABC']
            
            # Форматування
            display_df['Ціна (грн)'] = display_df['Ціна (грн)'].apply(lambda x: f"{x:.2f}")
            display_df['Продажі мережі'] = display_df['Продажі мережі'].apply(lambda x: f"{int(x):,}")
            display_df['Продажі магазину'] = display_df['Продажі магазину'].apply(lambda x: f"{int(x):,}")
            display_df['Потенціал (шт/міс)'] = display_df['Потенціал (шт/міс)'].apply(lambda x: f"{x:.1f}")

            # Кольорове виділення ABC
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

            # Метрики по рекомендаціях
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Рекомендовано товарів", len(filtered_rec))
            with col2:
                st.metric("Потенціал (шт/міс)", f"{filtered_rec['Potential_Qty'].sum():.0f}")
            with col3:
                st.metric("Потенціал (грн/міс)", f"{filtered_rec['Potential_Sum'].sum():,.0f}")
        else:
            st.info("Немає товарів, що відповідають обраним фільтрам")
    else:
        st.info("Рекомендації не знайдено. Спробуйте змінити параметри.")

    st.divider()

    # ABC аналіз
    st.subheader("📊 ABC Аналіз сегмента")
    
    if not abc_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Графік ABC
            abc_counts = abc_df['ABC'].value_counts().reindex(['A', 'B', 'C'], fill_value=0)
            fig_abc = px.pie(
                values=abc_counts.values,
                names=abc_counts.index,
                title="Розподіл товарів за категоріями ABC",
                color=abc_counts.index,
                color_discrete_map={'A': '#90EE90', 'B': '#FFE4B5', 'C': '#FFB6C1'}
            )
            st.plotly_chart(fig_abc, use_container_width=True)

        with col2:
            st.markdown("**Категорії ABC:**")
            for category in ['A', 'B', 'C']:
                count = len(abc_df[abc_df['ABC'] == category])
                revenue = abc_df[abc_df['ABC'] == category]['Revenue'].sum()
                pct = (revenue / abc_df['Revenue'].sum() * 100) if abc_df['Revenue'].sum() > 0 else 0
                st.metric(f"Категорія {category}", f"{count} товарів", f"{pct:.1f}% виручки")
    else:
        st.info("Недостатньо даних для ABC аналізу")

    st.divider()

    # Життєвий цикл
    st.subheader("🔄 Аналіз життєвого циклу товарів")
    
    if not lifecycle_df.empty:
        # Метрики по стадіях
        stage_summary = lifecycle_df['Stage'].value_counts()
        col1, col2, col3, col4 = st.columns(4)

        stages = ['Впровадження', 'Зростання', 'Зрілість', 'Спад']
        icons = ['🚀', '📈', '⚖️', '📉']
        cols = [col1, col2, col3, col4]

        for i, (stage, icon) in enumerate(zip(stages, icons)):
            with cols[i]:
                count = stage_summary.get(stage, 0)
                st.metric(f"{icon} {stage}", count)

        # Графік розподілу
        fig_lifecycle = px.pie(
            values=stage_summary.values,
            names=stage_summary.index,
            title="Розподіл товарів за стадіями життєвого циклу"
        )
        st.plotly_chart(fig_lifecycle, use_container_width=True)

        # Таблиця товарів
        with st.expander("📋 Детальна інформація по товарах"):
            lifecycle_display = lifecycle_df[['Art', 'Describe', 'Stage', 'Total_Sales',
                                             'Months_Active', 'Avg_Monthly_Sales']].copy()
            lifecycle_display.columns = ['Артикул', 'Опис', 'Стадія', 'Всього продажів',
                                        'Місяців активності', 'Середні продажі/міс']
            st.dataframe(lifecycle_display, use_container_width=True)
    else:
        st.info("Недостатньо даних для аналізу життєвого циклу")

    st.divider()

    # Сезонність
    st.subheader("📅 Аналіз сезонності продажів")

    if seasonality_data['peak_month'] != 'Немає даних':
        # Графік продажів по місяцях
        fig_season = go.Figure()

        fig_season.add_trace(go.Scatter(
            x=seasonality_data['months'],
            y=seasonality_data['sales'],
            mode='lines+markers',
            name='Продажі',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        fig_season.update_layout(
            title="Сезонність продажів в сегменті",
            xaxis_title="Місяць",
            yaxis_title="Кількість продажів",
            hovermode='x unified'
        )

        st.plotly_chart(fig_season, use_container_width=True)

        # Індекс сезонності
        fig_index = go.Figure()

        fig_index.add_trace(go.Bar(
            x=seasonality_data['months'],
            y=seasonality_data['seasonality_index'],
            marker_color=['#90EE90' if x >= 100 else '#FFB6C1' for x in seasonality_data['seasonality_index']],
            text=[f"{x:.0f}" for x in seasonality_data['seasonality_index']],
            textposition='outside'
        ))

        fig_index.add_hline(y=100, line_dash="dash", line_color="gray",
                           annotation_text="Середнє значення")

        fig_index.update_layout(
            title="Індекс сезонності (середнє = 100)",
            xaxis_title="Місяць",
            yaxis_title="Індекс",
            showlegend=False
        )

        st.plotly_chart(fig_index, use_container_width=True)

        # Інформація про піки
        col1, col2 = st.columns(2)
        with col1:
            peak_value = seasonality_data['sales'][seasonality_data['months'].index(seasonality_data['peak_month'])]
            st.success(f"📈 **Піковий місяць:** {seasonality_data['peak_month']} ({int(peak_value)} шт)")
        with col2:
            low_value = seasonality_data['sales'][seasonality_data['months'].index(seasonality_data['low_month'])]
            st.info(f"📉 **Низький місяць:** {seasonality_data['low_month']} ({int(low_value)} шт)")
    else:
        st.info("Недостатньо даних для аналізу сезонності")

    st.divider()

    # Детальна статистика
    st.subheader("📈 Детальна статистика")

    stats_table = create_statistics_table(df, store, segment, recommendations, abc_df, lifecycle_df)

    if not stats_table.empty:
        categories = stats_table['Категорія'].unique().tolist()
        
        if len(categories) > 0:
            tabs = st.tabs(categories)
            
            for i, category in enumerate(categories):
                with tabs[i]:
                    category_data = stats_table[stats_table['Категорія'] == category]
                    category_display = category_data[['Показник', 'Значення']].copy()

                    # Красиве відображення таблиці
                    st.dataframe(
                        category_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Показник": st.column_config.TextColumn("Показник", width="medium"),
                            "Значення": st.column_config.TextColumn("Значення", width="medium")
                        }
                    )
        else:
            st.info("Немає даних для відображення статистики")

def main():
    # Заголовок
    st.title("🛍️ Рекомендаційна система товарів, яка пропонує магазину перелік товарів, які добре продаються в мережі, але ще не представлені в даному магазині")
    st.markdown("""
    Система аналізу та рекомендацій товарів з використанням:
    - **ABC аналіз** - класифікація за виручкою
    - **Аналіз життєвого циклу** - визначення стадії товару
    - **Аналіз сезонності** - виявлення сезонних патернів
    - **Інтелектуальні алерти** - сповіщення про важливі події
    """)

    # Вибір джерела даних
    st.subheader("📥 Джерело даних")
    data_source = st.radio(
        "Оберіть джерело даних:",
        options=["Локальний файл", "Google Sheets"],
        horizontal=True
    )

    df = None

    if data_source == "Локальний файл":
        # Завантаження файлу
        uploaded_file = st.file_uploader(
            "📁 Завантажте Excel файл з даними про продажі",
            type=['xlsx', 'xls'],
            help="Файл повинен містити колонки: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum"
        )

        if uploaded_file is None:
            st.info("👆 Завантажте Excel файл для початку роботи")
            with st.expander("ℹ️ Вимоги до формату даних"):
                st.markdown("""
                **Обов'язкові колонки:**
                - `Magazin` - назва магазину
                - `Datasales` - дата продажу
                - `Art` - артикул товару
                - `Describe` - опис товару
                - `Model` - модель товару
                - `Segment` - сегмент товару
                - `Price` - ціна
                - `Qty` - кількість
                - `Sum` - сума продажу

                **Формати дат:** DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD
                """)
            return

        # Завантаження даних з файлу
        with st.spinner("⏳ Завантаження та обробка даних..."):
            df = load_and_process_data(uploaded_file)

    else:  # Google Sheets
        # Ініціалізація session_state для зберігання даних
        if 'google_sheets_data' not in st.session_state:
            st.session_state.google_sheets_data = None

        # Поле для введення URL
        sheet_url = st.text_input(
            "URL Google Sheets:",
            value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit?gid=64159818#gid=64159818",
            help="Таблиця має бути відкрита для перегляду (публічний доступ)"
        )

        if not sheet_url:
            st.info("👆 Введіть URL таблиці Google Sheets для початку роботи")
            with st.expander("ℹ️ Як отримати публічний доступ до таблиці?"):
                st.markdown("""
                **Інструкція:**
                1. Відкрийте вашу таблицю в Google Sheets
                2. Натисніть кнопку "Налаштування доступу" (праворуч вгорі)
                3. Оберіть "Всі, у кого є посилання"
                4. Встановіть права "Читач"
                5. Скопіюйте посилання на таблицю

                **Обов'язкові колонки:**
                - `Magazin`, `Datasales`, `Art`, `Describe`, `Model`, `Segment`, `Price`, `Qty`, `Sum`

                **Формати дат:** DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD
                """)
            return

        # Кнопка для завантаження даних
        load_button = st.button("📊 Завантажити дані з Google Sheets", type="primary")

        if load_button:
            with st.spinner("⏳ Завантаження та обробка даних з Google Sheets..."):
                loaded_df = load_data_from_google_sheets(sheet_url)
                if loaded_df is not None:
                    st.session_state.google_sheets_data = loaded_df
                    st.rerun()

        # Перевірка наявності завантажених даних
        if st.session_state.google_sheets_data is None:
            st.info("👆 Натисніть кнопку для завантаження даних")
            return

        # Використання даних з session_state
        df = st.session_state.google_sheets_data
    
    if df is None:
        return

    st.success(f"✅ Дані успішно завантажено: {len(df):,} записів")

    # Інформація про дані
    with st.expander("📊 Інформація про завантажені дані"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Записів", f"{len(df):,}")
        with col2:
            st.metric("Магазинів", df['Magazin'].nunique())
        with col3:
            st.metric("Товарів", df['Art'].nunique())
        with col4:
            st.metric("Сегментів", df['Segment'].nunique())

        date_range = f"{df['Datasales'].min().strftime('%d.%m.%Y')} - {df['Datasales'].max().strftime('%d.%m.%Y')}"
        st.info(f"📅 Період даних: {date_range}")

    # Бічна панель з параметрами
    with st.sidebar:
        st.header("⚙️ Параметри аналізу")

        # Вибір магазину та сегмента
        stores = sorted(df['Magazin'].unique())
        segments = sorted(df['Segment'].unique())

        selected_store = st.selectbox(
            "🏪 Оберіть магазин:",
            stores,
            help="Магазин для аналізу"
        )

        selected_segment = st.selectbox(
            "📊 Оберіть сегмент:",
            segments,
            help="Сегмент товарів"
        )

        st.divider()

        # Критерії рекомендацій
        st.subheader("🎯 Критерії рекомендацій")

        min_network_qty = st.number_input(
            "Мінімальні продажі в мережі:",
            min_value=1,
            max_value=1000,
            value=10,
            step=5,
            help="Мінімальна кількість продажів товару в мережі для рекомендації"
        )

        max_store_qty = st.number_input(
            "Максимальні продажі в магазині:",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            help="Максимальна кількість продажів в магазині (0 = товар відсутній)"
        )

        st.divider()

        # Кнопка аналізу
        analyze_btn = st.button(
            "🎯 Запустити аналіз",
            type="primary",
            use_container_width=True
        )
    
    # Аналіз
    if analyze_btn:
        with st.spinner("🔍 Виконується аналіз даних..."):
            try:
                # Генерація рекомендацій
                recommendations = generate_recommendations_with_abc(
                    df, selected_store, selected_segment, min_network_qty, max_store_qty
                )

                # ABC аналіз
                abc_df = calculate_abc_analysis(df, selected_segment)

                # Сезонність
                seasonality_data = calculate_seasonality(df, selected_segment)

                # Життєвий цикл
                lifecycle_df = analyze_product_lifecycle(df, selected_segment)

                # Алерти
                alerts = generate_alerts(df, selected_store, selected_segment, recommendations)

                # Відображення результатів
                st.success("✅ Аналіз завершено успішно!")
                st.divider()

                display_results(
                    df, selected_store, selected_segment,
                    recommendations, seasonality_data,
                    lifecycle_df, alerts, abc_df
                )

                # Експорт звіту
                st.divider()
                st.subheader("📥 Експорт звіту")

                excel_report = create_excel_report(
                    df, selected_store, selected_segment,
                    recommendations, abc_df, seasonality_data,
                    lifecycle_df, alerts
                )

                filename = f"analysis_report_{selected_store}_{selected_segment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                st.download_button(
                    label="📊 Завантажити повний звіт Excel",
                    data=excel_report.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Помилка при аналізі: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()


