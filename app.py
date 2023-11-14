import pandas as pd
import os
import re
from statsmodels.tsa.arima.model import ARIMA
import joblib
import requests
from datetime import datetime, timedelta
from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask import Flask
import numpy as np
import csv

app = Flask(__name__)

svm_classifier = joblib.load('svm_classifier_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_model.joblib')

def load_and_preprocess_data(filename):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['Close'] = data['Close'].astype(float)
    data = data.asfreq('B')
    return data

def clean_summary(text):
    cleaned_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    return cleaned_text.strip()


#--------CUSTOM BUY SELL SIGNALS-------
def buying_signals(data):
    # Volume Breakout
    data['Avg_Volume'] = data['Volume'].rolling(window=5).mean()
    data['Volume_Breakout'] = (data['Volume'] > data['Avg_Volume']) & (data['Close'] > data['Open'])

    # High Volume on Green Candles
    data['Green_Candle'] = data['Close'] > data['Open']
    data['High_Volume_Green'] = (data['Volume'] > data['Avg_Volume']) & data['Green_Candle']

    # Decreasing Volume during Pullbacks
    data['Pullback'] = data['Close'] < data['Close'].shift(1)
    data['Decreasing_Volume'] = data['Volume'] < data['Volume'].shift(1)
    data['Buy_Signal_Pullback'] = data['Pullback'] & data['Decreasing_Volume']

    # Modify 'Buy_Signal_Pullback' values
    data['Buy_Signal_Pullback'] = data['Buy_Signal_Pullback'].map({True: 'BUY', False: 'SELL', None: '--'}).fillna('--')

    return data['Buy_Signal_Pullback'].iloc[0]

# --------------VWAP SIGNAL CODE ------------
def vwap_signal(data):
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

    def below_vwap_count(row):
        count = 0
        if row['Open'] < row['VWAP']:
            count += 1
        if row['Close'] < row['VWAP']:
            count += 1
        if row['Low'] < row['VWAP']:
            count += 1
        return count

    def above_vwap_count(row):
        count = 0
        if row['Open'] > row['VWAP']:
            count += 1
        if row['Close'] > row['VWAP']:
            count += 1
        if row['High'] > row['VWAP']:
            count += 1
        return count

    data['BelowVWAPCount'] = data.apply(below_vwap_count, axis=1)
    data['AboveVWAPCount'] = data.apply(above_vwap_count, axis=1)

    data['VWAP_Signal'] = np.where(
        (data['BelowVWAPCount'] >= 2) &
        (data['High'] > data['Low'].shift(1)) &
        (data['Close'] > data['Low'].shift(1)), 'BUY',
        np.where(
            (data['AboveVWAPCount'] >= 2) &
            (data['Low'] < data['High'].shift(1)) &
            (data['Close'] < data['High'].shift(1)), 'SELL', '--'))

    return data['VWAP_Signal'].iloc[0]


# ----- VWMA Signal Code -----
def vwma_signal(data):
    length = 4
    vwma = (data['Close'] * data['Volume']).rolling(window=length).mean() / data['Volume'].rolling(window=length).mean()
    roc_vwma = vwma - vwma.shift(1)
    roc_sma_vwma = vwma.rolling(window=4).mean() - vwma.rolling(window=4).mean().shift(1)
    roc_of_roc_sma_vwma = roc_sma_vwma - roc_sma_vwma.shift(1)

    data['VWMA_Signal'] = ['BUY' if roc_vwma.iloc[idx] > roc_of_roc_sma_vwma.iloc[idx] else 'HOLD' for idx in data.index]

    return data['VWMA_Signal'].iloc[0]


# ----- Fibonacci Signal Code -----

def compute_fibo_recom(data):
    length = 8
    data['Level_1'] = data['High'].rolling(window=length).max()
    data['Level_0'] = data['Low'].rolling(window=length).min()
    data['Range'] = data['Level_1'] - data['Level_0']
    data['Level_786'] = data['Level_1'] - data['Range'] * 0.786
    data['Buy_condition'] = (data['Low'] <= data['Level_786']) & (data['Close'] > data['Open'])
    data['Fibonacci_Signal'] = data.apply(lambda row: 'BUY' if row['Buy_condition'] else 'HOLD', axis=1)

    return data['Fibonacci_Signal'].iloc[0]

import requests
from datetime import datetime, timedelta
import joblib

def get_news_recommendation(company_name):
    subscription_key = "07068875780c4a74bad61bddbede2826"
    company_name_abbreviation = ''.join([word[:3] for word in company_name.split()]).upper()

    recommendation_meanings = {
        0: 'Sell',
        1: 'Buy',
        2: 'Hold'
    }
    endpoint = f"https://api.bing.microsoft.com/v7.0/news/search?q={company_name}&category=Business&count=200&offset=0&mkt=en-in&safeSearch=Moderate&textFormat=Raw&textDecorations=false"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    response = requests.get(endpoint, headers=headers)
    news = response.json()

    # Relevant news filtering and formatting from the first code
    allowed_websites = ["moneycontrol.com", "cnbctv18.com", "livemint.com", "ndtv.com", "economictimes",
                        "business-standard.com", "reuters.com", "bloomberg.com", "marketwatch.com", "finance.yahoo.com",
                        "forbes.com", "moneycrashers.com", "investors.com", "investing.com", "reuters.com",
                        "mtnewswires.com", "djnewswires.com"]

    relevant_articles = [
        article for article in news['value']
        if company_name.lower() in article['name'].lower()
           and any(website in article['url'] for website in allowed_websites)
    ]

    if len(relevant_articles) <= 3:
        relevant_articles = [
            article for article in news['value']
            if company_name.lower() in article['name'].lower()
        ]

    final_articles = []

    for article in relevant_articles:
        if ('about' in article and any('share price' in about['name'].lower() for about in article['about'])) \
                or 'price' in article['name'].lower():
            continue

        if len(article['name'].split()) < 8:
            continue

        article_info = {
            'headline': article['name'],
            'url': article['url'],
            'datePosted': article['datePublished']
        }

        if 'image' in article and 'thumbnail' in article['image']:
            article_info['thumbnail'] = article['image']['thumbnail']['contentUrl']

        final_articles.append(article_info)

    headlines = [article['headline'] for article in final_articles]
    tfidf_matrix = tfidf_vectorizer.transform(headlines)
    predicted_recommendations = svm_classifier.predict(tfidf_matrix)

    buy_count = (predicted_recommendations == 1).sum()
    sell_count = (predicted_recommendations == 0).sum()
    hold_count = (predicted_recommendations == 2).sum()

    total_count = len(final_articles)
    buy_percentage = (buy_count / total_count) * 100 if total_count else 0
    sell_percentage = (sell_count / total_count) * 100 if total_count else 0
    hold_percentage = (hold_count / total_count) * 100 if total_count else 0

    max_percentage = max(buy_percentage, sell_percentage, hold_percentage)
    recommendation = "BUY" if max_percentage == buy_percentage else (
        "SELL" if max_percentage == sell_percentage else "HOLD")

    # Formatting the articles for return
    formatted_articles = []
    for article in final_articles:
        date_published = datetime.strptime(article['datePosted'], '%Y-%m-%dT%H:%M:%S.%f0Z')
        article['date'] = date_published.strftime('%Y-%m-%d')
        article['time'] = date_published.strftime('%H:%M')
        del article['datePosted']  # No need to return the original datePosted
        formatted_articles.append(article)

    return recommendation, news, formatted_articles


def drn_shortest_rsi(df, ema1_length=3, ema2_length=5, ema3_length=8):
    df['a'] = (df['High'] - df['Low'] + df['Close'] + df['Open']).ewm(span=ema1_length, adjust=False).mean() - \
              (df['High'] - df['Low'] + df['Close'] + df['Open']).ewm(span=ema2_length, adjust=False).mean() + \
              (df['High'] - df['Low'] + df['Close'] + df['Open']).ewm(span=ema3_length, adjust=False).mean()

    df['is_prev_smaller'] = df['a'] > df['a'].shift(-1)
    df['result'] = df.apply(lambda row: 'positive' if row['is_prev_smaller'] else 'negative', axis=1)

    recommendations = []
    for i in range(len(df) - 1):
        if df['result'].iloc[i] == "positive" and df['result'].iloc[i + 1] == "positive":
            recommendations.append("BUY")
        elif df['result'].iloc[i] == "negative" and df['result'].iloc[i + 1] == "negative":
            recommendations.append("SELL")
        else:
            recommendations.append("WAIT-WAIT")
    recommendations.append("--")
    df['recommendations'] = recommendations

    return df['recommendations'].iloc[0]

def generate_summary(news):
    headlines = [article['name'] for article in news['value']]
    combined_titles = " ".join(headlines)
    summary = clean_summary(combined_titles)
    lines = summary.split('. ')
    if len(lines) > 1 and len(lines[-1].split()) < 10:
        summary = '. '.join(lines[:-1]) + '.'
    return summary

def final_recommendation(model_a_output, model_b_output, model_c_output,model_d_output,model_e_output,model_f_output):
    buy_count = [model_a_output, model_b_output, model_c_output,model_d_output,model_e_output,model_f_output,].count("BUY")
    sell_count = [model_a_output, model_b_output, model_c_output,model_d_output,model_e_output,model_f_output].count("SELL")
    other_count = [model_a_output, model_b_output, model_c_output,model_d_output,model_e_output,model_f_output,].count("WAIT-WAIT") + [model_a_output, model_b_output,
                                                                                         model_c_output].count("HOLD")+[model_d_output,model_e_output,model_f_output].count("--")

    if buy_count >= 2:
        return "BUY"
    elif sell_count >= 2:
        return "SELL"
    elif buy_count >= 1 and (other_count == 1 or sell_count == 1):
        return "SLIGHTLY BUY"
    elif sell_count >= 1 and (other_count == 1 or buy_count == 1):
        return "SLIGHTLY SELL"
    else:
        return "NO COMMENTS"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return 'File uploaded and saved'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/recommendation', methods=['POST'])
def recommendation():
    try:
        if 'csv_file' not in request.files:
            return render_template('error.html', message="No file provided!")

        file = request.files['csv_file']
        company_name = request.form['company_name']

        if file.filename == '':
            return render_template('error.html', message="No file selected for uploading!")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            direction_svm, news, formatted_articles = get_news_recommendation(company_name)
            df = pd.read_csv(filepath)
            top_row_recommendation = drn_shortest_rsi(df)
            custom_recom = buying_signals(df)
            vwap_recom = vwap_signal(df)
            fibo_recom = compute_fibo_recom(df)

            vwma_recom = vwma_signal(df)
            final_recommendation_output = final_recommendation(direction_svm, top_row_recommendation,custom_recom,fibo_recom,vwap_recom,vwma_recom)

            summary = generate_summary(news)

            # Delete the saved file after processing
            os.remove(filepath)

            return render_template('index.html',
                                   direction_svm=direction_svm,
                                   top_row_recommendation=top_row_recommendation,
                                   custom_recom=custom_recom,
                                   vwap_recom=vwap_recom,
                                   fibo_recom=fibo_recom,
                                   vwma_recom=vwma_recom,
                                   final_recommendation_output=final_recommendation_output,
                                   summary=summary,
                                   articles=formatted_articles)

        return render_template('error.html', message="Invalid file type!")
    except Exception as e:
        return render_template('error.html', message=str(e))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)


