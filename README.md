# -Stock-Prediction-Flask-App
Developed a full-stack Flask web app for 5-day stock prediction. Features a unique ARIMAX-LSTM hybrid ML model integrating historical data, market breadth, and news sentiment from web scraping. Built with Python, TensorFlow/Keras, Pandas, and styled with Tailwind CSS. Offers an interactive, UI for dynamic stock search and visual forecasts.
Features
Hybrid Prediction Model: Utilizes a unique ARIMAX-LSTM architecture to capture both statistical trends and complex non-linear relationships in financial data for enhanced prediction accuracy.

External Factor Integration: Incorporates real-time market breadth and news sentiment into the prediction model to provide a more holistic forecast.

Historical Data & News Acquisition: Fetches historical stock data using yfinance and dynamically scrapes news headlines from financial sources (with robust error handling for API changes).

Interactive & Aesthetic Frontend: A user-friendly, dark-themed web interface built with HTML, CSS (Tailwind CSS), and JavaScript for a modern user experience.

Dynamic Stock Search & Selection: Allows users to search for stocks/indices and provides a list of matching options for precise selection, similar to professional trading platforms.

Visual Forecasts: Generates and displays clear, intuitive plots showing historical price movements alongside the 5-day future predictions.

Robust Error Handling: Implements server-side and client-side error handling to provide informative feedback to the user and ensure application stability.

Tech Stack
Backend (Python Flask Application):
Python: Core programming language.

Flask: Web framework for building the API and serving web pages.

Pandas: Data manipulation and analysis, especially for time series.

NumPy: Numerical computing.

Matplotlib: Plot generation for historical and predicted prices.

Scikit-learn: Data preprocessing (MinMaxScaler).

TensorFlow/Keras: Deep learning framework for LSTM model.

Statsmodels: Statistical modeling for ARIMAX model.

yfinance: Reliable historical market data.

yahoo_fin: (Attempted) News headlines, with fallbacks due to API changes.

Requests: HTTP requests for web scraping.

Gunicorn: (Recommended for production) WSGI HTTP server.

Frontend (Web Browser):
HTML5: Structure of the web page.

CSS3: Styling.

Tailwind CSS: Utility-first CSS framework for rapid UI development.

JavaScript (ES6+): Client-side interactivity, asynchronous requests (fetch API), and dynamic UI updates.

How to Run Locally
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Git

Clone the Repository First, clone the project repository from GitHub:
git clone https://github.com/WishchalSingh-Sudo/Stock-Prediction-Flask-App.git cd Stock-Prediction-Flask-App

Set Up Virtual Environment (Recommended) It's highly recommended to use a virtual environment to manage dependencies:
Create a virtual environment
python -m venv venv

Activate the virtual environment
On Windows:
.\venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

Install Dependencies Install all required Python packages:
pip install -r requirements.txt

Run the Flask Application Start the Flask development server:
python app.py

You should see output similar to this:

Running on http://127.0.0.1:8000/ (Press CTRL+C to quit)
Access the Application Open your web browser and navigate to the address provided by Flask, usually:
http://127.0.0.1:8000/

You can now use the application to get stock price predictions!

Deployment Considerations (Future Steps)
For production deployment, consider:

Disabling Flask debug mode.

Using a production-ready WSGI server like Gunicorn.

Deploying to platforms like Heroku, Render, AWS Elastic Beanstalk, or DigitalOcean.

Optimizing model training (e.g., training models offline and loading them on startup) to improve response times for concurrent users.
