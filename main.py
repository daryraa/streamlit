import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.ticker import FuncFormatter

# Judul dan Deskripsi
st.title('Analisis Data Retail')
st.write('Aplikasi ini menampilkan analisis data retail dengan RFM Analysis & Item Recomendation berbasis collaborative filtering .')

# Membaca dataset dari file Excel
df = pd.read_csv("data_input/retail100k.csv")

# Preprocessing
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Mengonversi kolom 'InvoiceNo' dan 'StockCode' ke string
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)

# Menambahkan kolom 'TotalPrice'
df.loc[:, 'TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Menghapus baris yang duplikat
df = df.drop_duplicates()

# Menambahkan tab
tabs = st.tabs(["Analisis Data Retail", "RFM", "Item Recommendation"])

with tabs[0]:
    # Bagian Analisis Data Retail
    st.subheader('Data Retail')
    st.write("Data Orginial" , df)
    st.write("Beberapa baris pertama dari dataset setelah preprocessing:", df)

    st.subheader('Pilih Negara:')
    unique_countries = df['Country'].unique().tolist()
    selected_countries = st.multiselect('Pilih Negara:', unique_countries, default=unique_countries[:5])
    filtered_df = df[df['Country'].isin(selected_countries)]

    # Visualisasi jumlah transaksi untuk negara yang dipilih
    st.subheader(f'Jumlah Transaksi di {selected_countries}')
    country_counts = filtered_df['Country'].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(12, 8))
    country_counts.plot(kind='bar', ax=ax)
    ax.set_title(f'Jumlah Transaksi di {selected_countries}')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Transactions')
    st.pyplot(fig)

    # Visualisasi top 10 produk yang paling sering dibeli
    st.subheader('Top 10 Produk yang Paling Sering Dibeli')
    top_10_products = df['Description'].value_counts().head(10)
    shortened_descriptions = top_10_products.index.str.slice(0, 15) + '...'
    shortened_top_10 = pd.Series(top_10_products.values, index=shortened_descriptions)
    fig, ax = plt.subplots(figsize=(12, 8))
    shortened_top_10.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Produk yang Paling Sering Dibeli')
    ax.set_xlabel('Product Description')
    ax.set_ylabel('Number of Times Purchased')
    ax.set_xticklabels(shortened_descriptions, rotation=45)
    st.pyplot(fig)

    # Mengonversi kolom 'InvoiceDate' ke tipe datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Menambahkan kolom 'Month', 'Day', dan 'Hour'
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.dayofweek
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Visualisasi jumlah transaksi berdasarkan bulan
    st.subheader('Jumlah Transaksi Berdasarkan Bulan')
    monthly_transactions = df.groupby('Month')['InvoiceNo'].nunique()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=monthly_transactions.index, y=monthly_transactions.values, marker='o', ax=ax)
    ax.set_title('Jumlah Transaksi Berdasarkan Bulan')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Transactions')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    st.pyplot(fig)

    # Visualisasi jumlah transaksi berdasarkan hari dalam seminggu
    st.subheader('Jumlah Transaksi Berdasarkan Hari dalam Seminggu')
    daily_transactions = df.groupby('Day')['InvoiceNo'].nunique()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=daily_transactions.index, y=daily_transactions.values, marker='o', ax=ax)
    ax.set_title('Jumlah Transaksi Berdasarkan Hari dalam Seminggu')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Transactions')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    st.pyplot(fig)

    # Fungsi untuk memformat nilai sumbu y
    def millions(x, pos):
        'The two args are the value and tick position'
        return '%1.1fM' % (x * 1e-6)

    # Menambahkan kolom 'MonthYear' untuk agregasi bulanan
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')

    # Menghitung total nilai transaksi bulanan
    monthly_transactions = df.groupby('MonthYear')['TotalPrice'].sum().reset_index()
    monthly_transactions['MonthYear'] = monthly_transactions['MonthYear'].dt.to_timestamp()

    # Visualisasi total nilai transaksi bulanan
    st.subheader('Total Nilai Transaksi Bulanan')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='MonthYear', y='TotalPrice', data=monthly_transactions, marker='o', ax=ax)
    ax.set_title('Total Nilai Transaksi Bulanan')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Total Nilai Transaksi')
    ax.yaxis.set_major_formatter(FuncFormatter(millions))
    st.pyplot(fig)

with tabs[1]:
    # Bagian RFM
    st.header('RFM Analysis')

    # Menentukan tanggal referensi untuk menghitung Recency
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    st.write("Reference Date: ", reference_date)

    # Menghitung Recency: Hari sejak terakhir kali pelanggan melakukan pembelian
    df['Recency'] = (reference_date - df['InvoiceDate']).dt.days

    # Menghitung nilai Recency untuk setiap pelanggan
    recency_df = df.groupby('CustomerID')['Recency'].min().reset_index()

    # Menghitung Frequency: Jumlah transaksi yang dilakukan oleh setiap pelanggan
    frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']

    # Menghitung Monetary: Total uang yang dihabiskan oleh setiap pelanggan
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    monetary_df = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary_df.columns = ['CustomerID', 'Monetary']

    # Menggabungkan semua metrik RFM ke dalam satu DataFrame
    rfm_df = recency_df.merge(frequency_df, on='CustomerID').merge(monetary_df, on='CustomerID')
    st.write("RFM DataFrame:", rfm_df)

    # Memberikan skor untuk masing-masing metrik RFM
    rfm_df['RecencyScore'] = pd.qcut(rfm_df['Recency'], 5, labels=False, duplicates='drop')
    rfm_df['FrequencyScore'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=False)
    rfm_df['MonetaryScore'] = pd.qcut(rfm_df['Monetary'], 5, labels=False)

    # Membuat skor RFM gabungan
    rfm_df['RFMScore'] = rfm_df['RecencyScore'].astype(str) + rfm_df['FrequencyScore'].astype(str) + rfm_df['MonetaryScore'].astype(str)

    st.write("RFM Score DataFrame:", rfm_df)

    # Menentukan segmen pelanggan berdasarkan skor RFM
    def segment_customer(df):
        if df['RFMScore'] == '444':
            return 'Best Customers'
        elif df['RFMScore'][0] == '4':
            return 'Loyal Customers'
        elif df['RFMScore'][1] == '4':
            return 'Frequent Customers'
        elif df['RFMScore'][2] == '4':
            return 'Big Spenders'
        elif df['RFMScore'][0] == '0':
            return 'Lost Customers'
        else:
            return 'Other'

    rfm_df['CustomerSegment'] = rfm_df.apply(segment_customer, axis=1)
    st.write("Customer Segments Count:", rfm_df['CustomerSegment'].value_counts())

    # Visualisasi segmen pelanggan
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='CustomerSegment', data=rfm_df, order=rfm_df['CustomerSegment'].value_counts().index, ax=ax)
    ax.set_title('Customer Segments Count')
    ax.set_xlabel('Customer Segment')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

with tabs[2]:
    # Bagian Item Recommendation
    st.header('Item Recommendation')
    st.write("Dataset Customer ID:", df[['CustomerID']])

    # Membuat matriks pivot dengan 'CustomerID' sebagai baris dan 'StockCode' sebagai kolom
    user_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum', fill_value=0)

    # Menghitung kesamaan kosinus antara pengguna
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Fungsi untuk merekomendasikan produk
    def recommend_products(customer_id, user_item_matrix, user_similarity_df, df, n_recommendations=5):
        # Mendapatkan skor kesamaan untuk pengguna yang diberikan
        similar_users = user_similarity_df[customer_id].sort_values(ascending=False)
        
        # Menghapus pengguna itu sendiri dari daftar kesamaan
        similar_users = similar_users.drop(customer_id)

        # Mengambil pengguna paling mirip
        top_similar_users = similar_users.head(n_recommendations).index

        # Menggabungkan pembelian pengguna mirip
        similar_users_purchases = user_item_matrix.loc[top_similar_users].sum().sort_values(ascending=False)

        # Menghapus produk yang sudah dibeli pengguna
        products_already_purchased = user_item_matrix.loc[customer_id]
        recommendations = similar_users_purchases.drop(products_already_purchased[products_already_purchased > 0].index)

        # Mendapatkan deskripsi produk
        recommended_products = recommendations.head(n_recommendations).index
        recommended_products_descriptions = df[df['StockCode'].isin(recommended_products)][['StockCode', 'Description']].drop_duplicates().set_index('StockCode')

        return recommended_products_descriptions

    # Input Customer ID
    customer_id = st.number_input('Masukkan Customer ID', min_value=0, step=1, format='%d')

    # Jika Customer ID diinputkan
    if customer_id in user_item_matrix.index:
        recommendations = recommend_products(customer_id, user_item_matrix, user_similarity_df, df)
        st.write("Rekomendasi produk untuk pelanggan {}:".format(customer_id))
        st.write(recommendations)
    else:
        st.write("Customer ID tidak ditemukan.")
