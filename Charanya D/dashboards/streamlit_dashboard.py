"""
Streamlit Dashboard for Retail Analytics
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Configure the page
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def make_api_request(endpoint: str):
    """Make API request with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def create_mock_data():
    """Create mock data for demonstration when API is not available"""
    return {
        'store_performance': [
            {'store_id': 'Mall_A', 'total_revenue': 125000, 'total_transactions': 1250, 'avg_transaction_value': 100},
            {'store_id': 'Mall_B', 'total_revenue': 98000, 'total_transactions': 980, 'avg_transaction_value': 100},
            {'store_id': 'Mall_C', 'total_revenue': 87000, 'total_transactions': 870, 'avg_transaction_value': 100},
        ],
        'customer_segments': {
            'Champions': 150,
            'Loyal Customers': 200,
            'Potential Loyalists': 250,
            'At Risk': 180,
            'New Customers': 220
        },
        'payment_methods': [
            {'method': 'Credit Card', 'percentage_share': 45.2, 'total_value': 75000},
            {'method': 'Debit Card', 'percentage_share': 24.1, 'total_value': 40000},
            {'method': 'Cash', 'percentage_share': 19.6, 'total_value': 32500},
            {'method': 'Digital Wallet', 'percentage_share': 10.8, 'total_value': 18000}
        ]
    }

# Main Dashboard
def main():
    st.title("🏪 Retail Store & Customer Insights Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section",
        ["Overview", "Store Performance", "Customer Analytics", "RFM Analysis", 
         "Profitability", "Seasonality", "Payment Methods", "Campaign Simulation"]
    )
    
    # Get data (try API first, fallback to mock data)
    mock_data = create_mock_data()
    
    if page == "Overview":
        show_overview(mock_data)
    elif page == "Store Performance":
        show_store_performance(mock_data)
    elif page == "Customer Analytics":
        show_customer_analytics(mock_data)
    elif page == "RFM Analysis":
        show_rfm_analysis()
    elif page == "Profitability":
        show_profitability_analysis()
    elif page == "Seasonality":
        show_seasonality_analysis()
    elif page == "Payment Methods":
        show_payment_methods(mock_data)
    elif page == "Campaign Simulation":
        show_campaign_simulation()

def show_overview(data):
    """Show overview dashboard"""
    st.header("📊 Business Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Revenue",
            value="$310K",
            delta="12.5%"
        )
    
    with col2:
        st.metric(
            label="Total Customers",
            value="1,000",
            delta="8.2%"
        )
    
    with col3:
        st.metric(
            label="Avg Order Value",
            value="$100",
            delta="5.1%"
        )
    
    with col4:
        st.metric(
            label="Customer Satisfaction",
            value="4.2/5",
            delta="0.3"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Store Performance Chart
        st.subheader("Store Performance")
        store_df = pd.DataFrame(data['store_performance'])
        fig = px.bar(store_df, x='store_id', y='total_revenue', 
                    title="Revenue by Store",
                    color='total_revenue',
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer Segments Pie Chart
        st.subheader("Customer Segments")
        segments_df = pd.DataFrame(list(data['customer_segments'].items()), 
                                 columns=['Segment', 'Count'])
        fig = px.pie(segments_df, values='Count', names='Segment',
                    title="Customer Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)

def show_store_performance(data):
    """Show store performance analytics"""
    st.header("🏬 Store Performance Analysis")
    
    # Store comparison
    store_df = pd.DataFrame(data['store_performance'])
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue Comparison")
        fig = px.bar(store_df, x='store_id', y='total_revenue',
                    title="Total Revenue by Store",
                    color='total_revenue',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Volume")
        fig = px.bar(store_df, x='store_id', y='total_transactions',
                    title="Number of Transactions by Store",
                    color='total_transactions',
                    color_continuous_scale='plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.subheader("Detailed Performance Metrics")
    store_df['efficiency_score'] = (store_df['total_revenue'] / store_df['total_transactions']).round(2)
    st.dataframe(store_df, use_container_width=True)
    
    # Insights
    st.subheader("Key Insights")
    best_store = store_df.loc[store_df['total_revenue'].idxmax(), 'store_id']
    st.success(f"🏆 {best_store} is the top performing store with ${store_df['total_revenue'].max():,} in revenue")
    
    avg_revenue = store_df['total_revenue'].mean()
    underperforming = store_df[store_df['total_revenue'] < avg_revenue]['store_id'].tolist()
    if underperforming:
        st.warning(f"⚠️ Stores below average: {', '.join(underperforming)}")

def show_customer_analytics(data):
    """Show customer analytics"""
    st.header("👥 Customer Analytics")
    
    # Customer segments
    segments_df = pd.DataFrame(list(data['customer_segments'].items()), 
                             columns=['Segment', 'Count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segmentation")
        fig = px.pie(segments_df, values='Count', names='Segment',
                    title="Customer Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Segment Analysis")
        fig = px.bar(segments_df, x='Segment', y='Count',
                    title="Customers per Segment",
                    color='Count',
                    color_continuous_scale='blues')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer insights
    st.subheader("Customer Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Value Customers", "150", "8.2%")
        st.info("Top 15% of customers by spend")
    
    with col2:
        st.metric("Average Customer Lifetime", "18 months", "2.1 months")
        st.info("Based on purchase frequency")
    
    with col3:
        st.metric("Customer Retention Rate", "75.5%", "3.2%")
        st.info("12-month retention rate")

def show_rfm_analysis():
    """Show RFM analysis"""
    st.header("📈 RFM Analysis")
    st.info("RFM stands for Recency, Frequency, and Monetary - key metrics for customer segmentation")
    
    # Mock RFM data
    rfm_data = {
        'Champions': {'count': 150, 'avg_recency': 15, 'avg_frequency': 25, 'avg_monetary': 2500},
        'Loyal Customers': {'count': 200, 'avg_recency': 25, 'avg_frequency': 18, 'avg_monetary': 1800},
        'Potential Loyalists': {'count': 250, 'avg_recency': 45, 'avg_frequency': 12, 'avg_monetary': 1200},
        'At Risk': {'count': 180, 'avg_recency': 120, 'avg_frequency': 8, 'avg_monetary': 950},
        'New Customers': {'count': 220, 'avg_recency': 30, 'avg_frequency': 3, 'avg_monetary': 400}
    }
    
    # Convert to DataFrame
    rfm_df = pd.DataFrame(rfm_data).T.reset_index()
    rfm_df.columns = ['Segment', 'Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
    
    # RFM Heatmap
    st.subheader("RFM Segment Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(rfm_df, x='Avg_Frequency', y='Avg_Monetary', 
                        size='Count', color='Segment',
                        title="Frequency vs Monetary Value",
                        hover_data=['Avg_Recency'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(rfm_df, x='Segment', y='Count',
                    title="Customer Count by RFM Segment",
                    color='Avg_Monetary',
                    color_continuous_scale='viridis')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM Table
    st.subheader("RFM Metrics by Segment")
    st.dataframe(rfm_df, use_container_width=True)
    
    # Action recommendations
    st.subheader("Recommended Actions")
    
    recommendations = {
        "Champions": "Reward them. They are your best customers - offer VIP treatment",
        "Loyal Customers": "Upsell higher value products. They are satisfied with purchases",
        "Potential Loyalists": "Offer membership/loyalty program to increase frequency",
        "At Risk": "Send personalized campaigns to re-engage them",
        "New Customers": "Provide onboarding support to improve early experience"
    }
    
    for segment, action in recommendations.items():
        st.info(f"**{segment}**: {action}")

def show_profitability_analysis():
    """Show profitability analysis"""
    st.header("💰 Profitability Analysis")
    
    # Mock profitability data
    categories = ['Clothing', 'Electronics', 'Books', 'Home & Garden', 'Sports']
    profits = [45000, 35000, 15000, 25000, 20000]
    margins = [38.5, 32.1, 28.9, 35.2, 33.7]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profit by Category")
        fig = px.bar(x=categories, y=profits, 
                    title="Total Profit by Category",
                    color=profits,
                    color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profit Margins")
        fig = px.bar(x=categories, y=margins,
                    title="Profit Margin % by Category",
                    color=margins,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Discount impact simulation
    st.subheader("Discount Impact Simulation")
    
    discount_rate = st.slider("Discount Rate (%)", 0, 30, 10, 5)
    
    original_revenue = 150000
    discounted_revenue = original_revenue * (1 - discount_rate/100)
    profit_impact = (original_revenue - discounted_revenue)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Revenue", f"${original_revenue:,}")
    
    with col2:
        st.metric("Revenue After Discount", f"${discounted_revenue:,}")
    
    with col3:
        st.metric("Revenue Impact", f"-${profit_impact:,}", f"-{discount_rate}%")

def show_seasonality_analysis():
    """Show seasonality analysis"""
    st.header("📅 Seasonality Analysis")
    
    # Mock seasonal data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sales_2023 = [80000, 70000, 90000, 95000, 100000, 120000,
                  130000, 125000, 100000, 95000, 140000, 150000]
    sales_2024 = [85000, 75000, 95000, 100000, 105000, 125000,
                  135000, 130000, 105000, 100000, 145000, 155000]
    
    # Create seasonal chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=sales_2023, mode='lines+markers', name='2023'))
    fig.add_trace(go.Scatter(x=months, y=sales_2024, mode='lines+markers', name='2024'))
    fig.update_layout(title="Monthly Sales Trends", xaxis_title="Month", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Peak Seasons")
        st.success("🎄 **Holiday Season** (Nov-Dec): Highest sales period")
        st.success("☀️ **Summer** (Jun-Aug): Second peak period")
        
    with col2:
        st.subheader("Low Seasons")
        st.warning("❄️ **Winter** (Jan-Feb): Lowest sales period")
        st.info("📈 **Growth Opportunity**: Focus marketing during low seasons")
    
    # Forecast
    st.subheader("Sales Forecast (Next 6 Months)")
    forecast_months = ['Jan 2025', 'Feb 2025', 'Mar 2025', 'Apr 2025', 'May 2025', 'Jun 2025']
    forecast_sales = [88000, 78000, 98000, 103000, 108000, 128000]
    
    fig = px.line(x=forecast_months, y=forecast_sales, 
                  title="Projected Sales Forecast",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

def show_payment_methods(data):
    """Show payment method analysis"""
    st.header("💳 Payment Method Analysis")
    
    payment_df = pd.DataFrame(data['payment_methods'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Payment Method Distribution")
        fig = px.pie(payment_df, values='percentage_share', names='method',
                    title="Payment Method Usage (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Value by Method")
        fig = px.bar(payment_df, x='method', y='total_value',
                    title="Total Value by Payment Method",
                    color='total_value',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment trends
    st.subheader("Payment Method Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Digital Payment Growth", "+25.5%", "YoY")
        st.info("Digital wallets and cards growing")
    
    with col2:
        st.metric("Cash Usage Decline", "-8.2%", "YoY")
        st.warning("Cash payments decreasing")
    
    with col3:
        st.metric("Average Digital Transaction", "$65", "+$5")
        st.success("Higher value digital transactions")

def show_campaign_simulation():
    """Show campaign simulation"""
    st.header("🎯 Campaign Simulation")
    
    st.subheader("Campaign Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_segment = st.selectbox(
            "Target Segment",
            ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "New Customers"]
        )
        
        discount_percentage = st.slider("Discount Percentage", 5, 30, 15, 5)
    
    with col2:
        campaign_budget = st.number_input("Campaign Budget ($)", 1000, 50000, 10000, 1000)
        
        campaign_type = st.selectbox(
            "Campaign Type",
            ["Email Marketing", "SMS Campaign", "Social Media Ads", "Direct Mail"]
        )
    
    # Simulate campaign results
    if st.button("🚀 Simulate Campaign"):
        # Mock simulation logic
        if target_segment == "Champions":
            estimated_reach = 150
            response_rate = 0.20
            avg_customer_value = 2500
        elif target_segment == "Loyal Customers":
            estimated_reach = 200
            response_rate = 0.18
            avg_customer_value = 1800
        elif target_segment == "At Risk":
            estimated_reach = 180
            response_rate = 0.12
            avg_customer_value = 950
        else:
            estimated_reach = 250
            response_rate = 0.15
            avg_customer_value = 1200
        
        expected_responders = int(estimated_reach * response_rate)
        gross_revenue = expected_responders * avg_customer_value
        discount_cost = gross_revenue * (discount_percentage / 100)
        net_revenue = gross_revenue - discount_cost - campaign_budget
        roi = (net_revenue / campaign_budget) * 100 if campaign_budget > 0 else 0
        
        st.subheader("📊 Campaign Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Estimated Reach", f"{estimated_reach:,}")
        
        with col2:
            st.metric("Expected Responders", f"{expected_responders:,}")
        
        with col3:
            st.metric("Projected Revenue", f"${gross_revenue:,}")
        
        with col4:
            st.metric("ROI", f"{roi:.1f}%", f"${net_revenue:,}")
        
        # ROI assessment
        if roi > 200:
            st.success("🎉 Excellent ROI! This campaign is highly recommended.")
        elif roi > 100:
            st.success("✅ Good ROI! This campaign should be profitable.")
        elif roi > 0:
            st.warning("⚠️ Positive but low ROI. Consider optimizing parameters.")
        else:
            st.error("❌ Negative ROI. This campaign may not be profitable.")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        if discount_percentage > 20:
            recommendations.append("Consider reducing discount rate to preserve margins")
        if roi < 100:
            recommendations.append("Try targeting a more responsive segment")
        if campaign_budget > gross_revenue * 0.2:
            recommendations.append("Campaign budget seems high relative to expected revenue")
        
        for rec in recommendations:
            st.info(f"• {rec}")

if __name__ == "__main__":
    main()
