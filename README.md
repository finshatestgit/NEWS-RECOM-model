Stock Recommendation System

This system is designed to analyze stock market data and news articles to provide a recommendation for buying, selling, or holding a particular stock.

 Key Features:

1. Stock Data Analysis: Uses various technical indicators like buying signals, VWAP, VWMA, Fibonacci, and more to assess the stock's trend.
2. News Analysis: It leverages Bing News API to fetch recent news articles about the company and assesses their sentiment using an SVM classifier.
3. Final Recommendation: Based on all the individual indicators, it computes a final recommendation.
4. Data Visualization: The Flask web application allows users to upload their stock data in CSV format and view the recommendation results.
5. Summarization: Using the BART model, the system provides a concise summary of the news articles.

 How to Use:

1. Setup:
   - Ensure all required libraries mentioned in the code are installed.
   - Download and set up Flask environment.
   - Obtain a subscription key from Bing News Search API and update the `subscription_key` variable.

2. Run the Application:
   - Run the provided Python code. The Flask server will start locally.
   - Open your web browser and navigate to `localhost:5000`.

3. Upload Stock Data:
   - On the main page, provide the company's name.
   - Upload a CSV file containing the stock data (Date, Open, High, Low, Close, Volume).
   - Click on the 'Submit' button.

4. View Recommendation:
   - The system will analyze the data and display individual recommendations from various models.
   - A final consolidated recommendation will also be displayed.
   - Additionally, a summary of recent news articles about the company will be provided.

 Note:

- Make sure the CSV file structure matches with the system's expectations.
- The Bing News API has a rate limit, be cautious about the number of requests.

 Credits:

This system utilizes libraries such as pandas, statsmodels, transformers, requests, and more. Special thanks to all the open-source contributors and libraries that made this project possible.
