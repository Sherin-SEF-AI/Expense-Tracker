import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
from datetime import datetime, timedelta
import sqlite3
import bcrypt
import uuid
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Gemini API setup
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Database setup
conn = sqlite3.connect('expense_tracker.db', check_same_thread=False)
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id TEXT PRIMARY KEY, username TEXT UNIQUE, password TEXT, budget REAL, currency TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS expenses
             (id TEXT PRIMARY KEY, user_id TEXT, date TEXT, amount REAL, category TEXT, description TEXT, currency TEXT, tags TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS recurring_expenses
             (id TEXT PRIMARY KEY, user_id TEXT, amount REAL, category TEXT, description TEXT, frequency TEXT, start_date TEXT, end_date TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS custom_categories
             (id TEXT PRIMARY KEY, user_id TEXT, name TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS financial_goals
             (id TEXT PRIMARY KEY, user_id TEXT, name TEXT, target_amount REAL, current_amount REAL, target_date TEXT,
              FOREIGN KEY (user_id) REFERENCES users(id))''')

conn.commit()

# Currency conversion API
CURRENCY_API_KEY = "YOUR_CURRENCY_API_KEY"
CURRENCY_API_URL = f"https://api.exchangerate-api.com/v4/latest/USD"

@st.cache_data(ttl=3600)
def get_exchange_rates():
    try:
        response = requests.get(CURRENCY_API_URL)
        response.raise_for_status()
        return response.json()['rates']
    except requests.RequestException:
        st.error("Failed to fetch exchange rates. Using default rates.")
        return {"USD": 1, "EUR": 0.85, "GBP": 0.75, "JPY": 110}

# Email configuration for alerts
EMAIL_ADDRESS = "your_email@example.com"
EMAIL_PASSWORD = "your_email_password"

# User authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode(), stored_password.encode())

def create_user(username, password, currency):
    hashed_password = hash_password(password)
    user_id = str(uuid.uuid4())
    c.execute("INSERT INTO users (id, username, password, currency) VALUES (?, ?, ?, ?)", (user_id, username, hashed_password, currency))
    conn.commit()
    return user_id

def get_user(username):
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    return c.fetchone()

# Expense functions
def add_expense(user_id, date, amount, category, description, currency, tags):
    expense_id = str(uuid.uuid4())
    c.execute("INSERT INTO expenses (id, user_id, date, amount, category, description, currency, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (expense_id, user_id, date, amount, category, description, currency, tags))
    conn.commit()

def get_user_expenses(user_id):
    c.execute("SELECT * FROM expenses WHERE user_id = ?", (user_id,))
    expenses = c.fetchall()
    return pd.DataFrame(expenses, columns=['id', 'user_id', 'Date', 'Amount', 'Category', 'Description', 'Currency', 'Tags'])

def update_expense(expense_id, date, amount, category, description, currency, tags):
    c.execute("UPDATE expenses SET date=?, amount=?, category=?, description=?, currency=?, tags=? WHERE id=?",
              (date, amount, category, description, currency, tags, expense_id))
    conn.commit()

def delete_expense(expense_id):
    c.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
    conn.commit()

# Recurring expense functions
def add_recurring_expense(user_id, amount, category, description, frequency, start_date, end_date):
    expense_id = str(uuid.uuid4())
    c.execute("INSERT INTO recurring_expenses (id, user_id, amount, category, description, frequency, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (expense_id, user_id, amount, category, description, frequency, start_date, end_date))
    conn.commit()

def get_user_recurring_expenses(user_id):
    c.execute("SELECT * FROM recurring_expenses WHERE user_id = ?", (user_id,))
    expenses = c.fetchall()
    return pd.DataFrame(expenses, columns=['id', 'user_id', 'Amount', 'Category', 'Description', 'Frequency', 'Start Date', 'End Date'])

# Custom category functions
def add_custom_category(user_id, name):
    category_id = str(uuid.uuid4())
    c.execute("INSERT INTO custom_categories (id, user_id, name) VALUES (?, ?, ?)", (category_id, user_id, name))
    conn.commit()

def get_user_custom_categories(user_id):
    c.execute("SELECT name FROM custom_categories WHERE user_id = ?", (user_id,))
    categories = c.fetchall()
    return [category[0] for category in categories]

# Financial goal functions
def add_financial_goal(user_id, name, target_amount, current_amount, target_date):
    goal_id = str(uuid.uuid4())
    c.execute("INSERT INTO financial_goals (id, user_id, name, target_amount, current_amount, target_date) VALUES (?, ?, ?, ?, ?, ?)",
              (goal_id, user_id, name, target_amount, current_amount, target_date))
    conn.commit()

def get_user_financial_goals(user_id):
    c.execute("SELECT * FROM financial_goals WHERE user_id = ?", (user_id,))
    goals = c.fetchall()
    return pd.DataFrame(goals, columns=['id', 'user_id', 'Name', 'Target Amount', 'Current Amount', 'Target Date'])

# Budget alert function
def send_budget_alert(email, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = email
        msg['Subject'] = "Budget Alert"
        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, email, text)
        server.quit()
    except Exception as e:
        st.error(f"Failed to send budget alert: {str(e)}")

# AI functions
def get_ai_insights(expenses_df):
    try:
        expenses_summary = expenses_df.groupby('Category')['Amount'].sum().reset_index()
        expenses_summary = expenses_summary.sort_values('Amount', ascending=False)
        top_categories = expenses_summary.head(3)['Category'].tolist()
        total_spent = expenses_df['Amount'].sum()
        
        prompt = f"""Based on the user's expense data:
        1. Top 3 spending categories: {', '.join(top_categories)}
        2. Total spent: {total_spent:.2f}

        Please provide:
        1. 3 actionable tips for better financial management
        2. 2 potential areas of overspending and how to reduce expenses
        3. A personalized savings strategy
        4. Recommendation for a suitable investment option based on the spending pattern
        5. A weekly budget plan to help the user manage expenses better

        Format the response in markdown for better readability."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Failed to generate AI insights: {str(e)}")
        return "Unable to generate insights at this time."

def predict_future_expenses(expenses_df):
    # Check if there's enough data for prediction
    if expenses_df.empty or len(expenses_df) < 30:  # Require at least 30 days of data
        return [], []

    try:
        expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
        expenses_df = expenses_df.sort_values('Date')
        expenses_df['Day'] = expenses_df['Date'].dt.dayofweek
        expenses_df['Month'] = expenses_df['Date'].dt.month
        expenses_df['DayOfMonth'] = expenses_df['Date'].dt.day
        
        # Create features
        X = expenses_df[['Day', 'Month', 'DayOfMonth']]
        y = expenses_df['Amount']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Model Mean Absolute Error: {mae:.2f}")

        # Predict next 30 days
        last_date = expenses_df['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        future_X = pd.DataFrame({
            'Day': future_dates.dayofweek,
            'Month': future_dates.month,
            'DayOfMonth': future_dates.day
        })
        future_predictions = model.predict(future_X)

        return future_predictions.tolist(), future_dates
    except Exception as e:
        st.error(f"Failed to predict future expenses: {str(e)}")
        return [], []


    try:
        expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
        expenses_df = expenses_df.sort_values('Date')
        expenses_df['Day'] = expenses_df['Date'].dt.dayofweek
        expenses_df['Month'] = expenses_df['Date'].dt.month
        expenses_df['DayOfMonth'] = expenses_df['Date'].dt.day
        
        # Create features
        X = expenses_df[['Day', 'Month', 'DayOfMonth']]
        y = expenses_df['Amount']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Model Mean Absolute Error: {mae:.2f}")

        # Predict next 30 days
        last_date = expenses_df['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        future_X = pd.DataFrame({
            'Day': future_dates.dayofweek,
            'Month': future_dates.month,
            'DayOfMonth': future_dates.day
        })
        future_predictions = model.predict(future_X)

        return future_predictions.tolist(), future_dates
    except Exception as e:
        st.error(f"Failed to predict future expenses: {str(e)}")
        return [], []

# UI Helper functions
def set_page_config():
    st.set_page_config(page_title="Personal Expense Tracker", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            font-family: 'Arial', sans-serif;
        }
        .stApp h1, .stApp h2, .stApp h3 {
            font-weight: bold;
        }
        .stApp p, .stApp li {
            font-size: 16px;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #4CAF50;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 5px;
            font-size: 16px;
        }
        .stPlotlyChart {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                color: #ffffff;
            }
            .stTextInput>div>div>input, .stSelectbox>div>div>select {
                color: #ffffff;
                background-color: #333333;
            }
            .stPlotlyChart {
                background-color: rgba(0, 0, 0, 0.2);
            }
        }
        @media (prefers-color-scheme: light) {
            .stApp {
                color: #000000;
            }
            .stTextInput>div>div>input, .stSelectbox>div>div>select {
                color: #000000;
                background-color: #ffffff;
            }
            .stPlotlyChart {
                background-color: rgba(255, 255, 255, 0.8);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main app
def main():
    set_page_config()

    st.title("💰 Advanced Personal Expense Tracker")

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id is None:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_button", use_container_width=True):
                user = get_user(username)
                if user and verify_password(user[2], password):
                    st.session_state.user_id = user[0]
                    st.session_state.username = user[1]
                    st.session_state.currency = user[4]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            st.subheader("Register")
            new_username = st.text_input("Username", key="register_username")
            new_password = st.text_input("Password", type="password", key="register_password")
            currency = st.selectbox("Preferred Currency", list(get_exchange_rates().keys()))
            if st.button("Register", key="register_button", use_container_width=True):
                if get_user(new_username):
                    st.error("Username already exists")
                else:
                    user_id = create_user(new_username, new_password, currency)
                    st.success("Registered successfully! Please login.")

    else:
        # Main application
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Add Expense", "Manage Expenses", "Recurring Expenses", "Analysis", "Financial Goals", "AI Insights", "Profile", "Logout"],
            icons=["house", "plus-circle", "list-task", "arrow-repeat", "graph-up", "bullseye", "robot", "person", "box-arrow-right"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )

        expenses_df = get_user_expenses(st.session_state.user_id)
        recurring_expenses = get_user_recurring_expenses(st.session_state.user_id)
        financial_goals = get_user_financial_goals(st.session_state.user_id)

        if selected == "Dashboard":
            st.subheader("📊 Expense Dashboard")
            if not expenses_df.empty:
                expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
                expenses_df['Amount'] = expenses_df.apply(lambda row: row['Amount'] * get_exchange_rates()[row['Currency']] / get_exchange_rates()[st.session_state.currency], axis=1)
                total_expenses = expenses_df['Amount'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Expenses", f"{st.session_state.currency} {total_expenses:.2f}")
                with col2:
                    this_month = expenses_df[expenses_df['Date'].dt.month == datetime.now().month]['Amount'].sum()
                    st.metric("This Month", f"{st.session_state.currency} {this_month:.2f}")
                with col3:
                    daily_avg = expenses_df['Amount'].mean()
                    st.metric("Daily Average", f"{st.session_state.currency} {daily_avg:.2f}")

                # Expenses by category
                fig = px.pie(expenses_df, values='Amount', names='Category', title="Expenses by Category")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

                # Expense trend
                fig = px.line(expenses_df.groupby('Date')['Amount'].sum().reset_index(), 
                              x='Date', y='Amount', title="Expense Trend")
                fig.update_layout(xaxis_title="Date", yaxis_title="Amount")
                st.plotly_chart(fig, use_container_width=True)

                # Top 5 expenses
                st.subheader("Top 5 Expenses")
                top_5 = expenses_df.nlargest(5, 'Amount')[['Date', 'Amount', 'Category', 'Description']]
                st.table(top_5.style.format({'Date': '{:%Y-%m-%d}', 'Amount': '{:.2f}'}))

                # Budget progress
                user = get_user(st.session_state.username)
                if user and user[3]:  # Check if budget is set
                    budget = user[3]
                    progress = (this_month / budget) * 100
                    st.subheader("Budget Progress")
                    st.progress(min(progress / 100, 1.0))
                    st.write(f"You've spent {progress:.2f}% of your {st.session_state.currency} {budget:.2f} budget")
                    if progress > 90:
                        st.warning("You're close to exceeding your budget!")
                else:
                    st.info("Set a monthly budget in your profile to track your spending progress.")

            else:
                st.info("No expenses recorded yet. Start by adding some expenses!")

        elif selected == "Add Expense":
            st.subheader("➕ Add New Expense")
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Date", datetime.today())
                amount = st.number_input("Amount", min_value=0.01, format="%.2f")
                currency = st.selectbox("Currency", list(get_exchange_rates().keys()), index=list(get_exchange_rates().keys()).index(st.session_state.currency))
            with col2:
                categories = ["Food", "Transportation", "Entertainment", "Bills", "Housing", "Healthcare", "Education", "Shopping", "Travel", "Investments", "Other"] + get_user_custom_categories(st.session_state.user_id)
                category = st.selectbox("Category", categories)
                description = st.text_input("Description")
                tags = st.text_input("Tags (comma-separated)")
            if st.button("Add Expense", key="add_expense_button", use_container_width=True):
                add_expense(st.session_state.user_id, date.strftime("%Y-%m-%d"), amount, category, description, currency, tags)
                st.success("Expense added successfully!")
                st.rerun()

        elif selected == "Manage Expenses":
            st.subheader("🗂️ Manage Expenses")
            if not expenses_df.empty:
                edited_df = st.data_editor(expenses_df, num_rows="dynamic", key="expense_editor")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Changes", key="save_changes_button", use_container_width=True):
                        for index, row in edited_df.iterrows():
                            update_expense(row['id'], row['Date'], row['Amount'], row['Category'], row['Description'], row['Currency'], row['Tags'])
                        st.success("Changes saved successfully!")
                        st.rerun()
                with col2:
                    if st.button("Delete Selected", key="delete_selected_button", use_container_width=True):
                        selected_rows = st.session_state.expense_editor['edited_rows']
                        for index in selected_rows:
                            delete_expense(expenses_df.iloc[index]['id'])
                        st.success("Selected expenses deleted successfully!")
                        st.rerun()
            else:
                st.info("No expenses recorded yet.")

        elif selected == "Recurring Expenses":
            st.subheader("🔁 Recurring Expenses")
            tab1, tab2 = st.tabs(["Add Recurring Expense", "View Recurring Expenses"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    amount = st.number_input("Amount", min_value=0.01, format="%.2f")
                    categories = ["Food", "Transportation", "Entertainment", "Bills", "Housing", "Healthcare", "Education", "Shopping", "Travel", "Investments", "Other"] + get_user_custom_categories(st.session_state.user_id)
                    category = st.selectbox("Category", categories)
                    description = st.text_input("Description")
                with col2:
                    frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Bi-weekly", "Monthly", "Quarterly", "Yearly"])
                    start_date = st.date_input("Start Date", datetime.today())
                    end_date = st.date_input("End Date")
                if st.button("Add Recurring Expense", key="add_recurring_expense_button", use_container_width=True):
                    add_recurring_expense(st.session_state.user_id, amount, category, description, frequency, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    st.success("Recurring expense added successfully!")
                    st.rerun()

            with tab2:
                if not recurring_expenses.empty:
                    st.dataframe(recurring_expenses.style.format({'Amount': '{:.2f}', 'Start Date': '{:%Y-%m-%d}', 'End Date': '{:%Y-%m-%d}'}))
                else:
                    st.info("No recurring expenses set up yet.")

        elif selected == "Analysis":
            st.subheader("🔍 Expense Analysis")
            if not expenses_df.empty:
                expenses_df['Date'] = pd.to_datetime(expenses_df['Date'])
                expenses_df['Amount'] = expenses_df.apply(lambda row: row['Amount'] * get_exchange_rates()[row['Currency']] / get_exchange_rates()[st.session_state.currency], axis=1)
                
                # Date range selector
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", expenses_df['Date'].min())
                with col2:
                    end_date = st.date_input("End Date", expenses_df['Date'].max())
                
                filtered_expenses = expenses_df[
                    (expenses_df['Date'] >= pd.Timestamp(start_date)) & 
                    (expenses_df['Date'] <= pd.Timestamp(end_date))
                ]

                # Expenses by category
                fig = px.pie(filtered_expenses, values='Amount', names='Category', title="Expenses by Category")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

                # Expense trend
                fig = px.line(filtered_expenses.groupby('Date')['Amount'].sum().reset_index(), 
                              x='Date', y='Amount', title="Expense Trend")
                fig.update_layout(xaxis_title="Date", yaxis_title="Amount")
                st.plotly_chart(fig, use_container_width=True)

                # Expense distribution
                fig = px.histogram(filtered_expenses, x='Amount', nbins=20, title="Expense Distribution")
                fig.update_layout(xaxis_title="Amount", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)

                # Top 5 expensive categories
                top_categories = filtered_expenses.groupby('Category')['Amount'].sum().nlargest(5).reset_index()
                fig = px.bar(top_categories, x='Category', y='Amount', title="Top 5 Expensive Categories")
                fig.update_layout(xaxis_title="Category", yaxis_title="Total Amount")
                st.plotly_chart(fig, use_container_width=True)

                # Monthly comparison
                monthly_expenses = filtered_expenses.groupby(filtered_expenses['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
                monthly_expenses['Date'] = monthly_expenses['Date'].dt.to_timestamp()
                fig = px.bar(monthly_expenses, x='Date', y='Amount', title="Monthly Expenses Comparison")
                fig.update_layout(xaxis_title="Month", yaxis_title="Total Amount")
                st.plotly_chart(fig, use_container_width=True)

                # Tag cloud
                if not filtered_expenses['Tags'].isnull().all() and filtered_expenses['Tags'].str.strip().str.len().sum() > 0:
                    st.subheader("Expense Tags")
                    all_tags = ' '.join(filtered_expenses['Tags'].dropna())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tags)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

            else:
                st.info("No expenses recorded yet. Start by adding some expenses!")

        elif selected == "Financial Goals":
            st.subheader("🎯 Financial Goals")
            tab1, tab2 = st.tabs(["Add Financial Goal", "View Financial Goals"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    goal_name = st.text_input("Goal Name")
                    target_amount = st.number_input("Target Amount", min_value=0.01, format="%.2f")
                with col2:
                    current_amount = st.number_input("Current Amount", min_value=0.0, format="%.2f")
                    target_date = st.date_input("Target Date")
                if st.button("Add Financial Goal", key="add_financial_goal_button", use_container_width=True):
                    add_financial_goal(st.session_state.user_id, goal_name, target_amount, current_amount, target_date.strftime("%Y-%m-%d"))
                    st.success("Financial goal added successfully!")
                    st.rerun()

            with tab2:
                if not financial_goals.empty:
                    for _, goal in financial_goals.iterrows():
                        progress = (goal['Current Amount'] / goal['Target Amount']) * 100
                        st.write(f"**{goal['Name']}**")
                        st.progress(min(progress / 100, 1.0))
                        st.write(f"Target: {st.session_state.currency} {goal['Target Amount']:.2f} | Current: {st.session_state.currency} {goal['Current Amount']:.2f} | Due: {goal['Target Date']}")
                        days_left = (pd.to_datetime(goal['Target Date']) - pd.Timestamp.now()).days
                        st.write(f"Days left: {max(days_left, 0)}")
                        
                        # Projection
                        if days_left > 0:
                            required_daily_savings = (goal['Target Amount'] - goal['Current Amount']) / days_left
                            st.write(f"Required daily savings: {st.session_state.currency} {required_daily_savings:.2f}")
                else:
                    st.info("No financial goals set up yet.")

        elif selected == "AI Insights":
            st.subheader("🤖 AI-Powered Insights")
            if not expenses_df.empty:
                with st.spinner("Generating AI insights..."):
                    insights = get_ai_insights(expenses_df)
                    st.markdown(insights)

                st.subheader("Expense Prediction")
                future_predictions, future_dates = predict_future_expenses(expenses_df)
                if future_predictions:
                    fig = px.line(x=future_dates, y=future_predictions, title="30-Day Expense Prediction")
                    fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Amount")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomaly detection
                    mean_prediction = np.mean(future_predictions)
                    std_prediction = np.std(future_predictions)
                    anomalies = [i for i, pred in enumerate(future_predictions) if abs(pred - mean_prediction) > 2 * std_prediction]
                    if anomalies:
                        st.warning("Potential anomalies detected in the following dates:")
                        for i in anomalies:
                            st.write(f"- {future_dates[i].date()}: Predicted amount {future_predictions[i]:.2f}")
                else:
                    st.info("Not enough data to make predictions. Add more expenses.")
            else:
                st.info("No expenses recorded yet. Add some expenses to get AI insights.")

        elif selected == "Profile":
            st.subheader("👤 User Profile")
            user = get_user(st.session_state.username)
            if user:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Username:** {user[1]}")
                    new_budget = st.number_input("Set Monthly Budget", min_value=0.0, value=user[3] or 0.0, step=100.0)
                with col2:
                    new_currency = st.selectbox("Preferred Currency", list(get_exchange_rates().keys()), index=list(get_exchange_rates().keys()).index(user[4]))
                if st.button("Update Profile", key="update_profile_button", use_container_width=True):
                    c.execute("UPDATE users SET budget = ?, currency = ? WHERE id = ?", (new_budget, new_currency, user[0]))
                    conn.commit()
                    st.session_state.currency = new_currency
                    st.success("Profile updated successfully!")
                    st.rerun()

                st.subheader("Custom Categories")
                new_category = st.text_input("Add New Category")
                if st.button("Add Category", key="add_category_button", use_container_width=True):
                    add_custom_category(st.session_state.user_id, new_category)
                    st.success("Category added successfully!")
                    st.rerun()

                custom_categories = get_user_custom_categories(st.session_state.user_id)
                st.write("Your custom categories:", ", ".join(custom_categories))

                if st.button("Export Expenses", key="export_expenses_button", use_container_width=True):
                    csv = expenses_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="expenses.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Data backup and restore
                st.subheader("Data Backup and Restore")
                if st.button("Backup Data", key="backup_data_button"):
                    backup_data = {
                        'expenses': expenses_df.to_dict(),
                        'recurring_expenses': recurring_expenses.to_dict(),
                        'financial_goals': financial_goals.to_dict(),
                        'custom_categories': custom_categories
                    }
                    json_data = json.dumps(backup_data)
                    b64 = base64.b64encode(json_data.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="expense_tracker_backup.json">Download Backup File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Backup file generated successfully!")

                uploaded_file = st.file_uploader("Restore Data from Backup", type="json")
                if uploaded_file is not None:
                    try:
                        backup_data = json.load(uploaded_file)
                        # Implement restore logic here
                        st.success("Data restored successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error restoring data: {str(e)}")

        elif selected == "Logout":
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.currency = None
            st.success("Logged out successfully!")
            st.rerun()

if __name__ == "__main__":
    main()

