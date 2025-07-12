import streamlit as st
import requests
import pandas as pd
import datetime
import pytz
import plotly.express as px
import plotly.graph_objects as go
from base64 import b64encode
import json
import urllib.parse
import os
import warnings

# eBay API credentials
CLIENT_ID = st.secrets["ebay"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["ebay"]["CLIENT_SECRET"]


# Encode credentials
credentials = b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()

# Get OAuth2 token
@st.cache_data(ttl=3600)
def get_access_token():
    token_url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {credentials}"
    }
    data = {
        "grant_type": "client_credentials",
        "scope": "https://api.ebay.com/oauth/api_scope"
    }
    response = requests.post(token_url, headers=headers, data=data)
    return response.json().get("access_token")

access_token = get_access_token()

# Seller categorization function
def categorize_seller(feedback_score, feedback_percent):
    try:
        score = int(feedback_score) if feedback_score is not None else 0
        percent = float(feedback_percent) if feedback_percent is not None else 0
    except (ValueError, TypeError):
        return "Uncategorized"
    
    if score >= 5000 and percent >= 99:
        return "Elite"
    elif score >= 1000 and percent >= 98:
        return "Excellent"
    elif score >= 500 and percent >= 97:
        return "Very Good"
    elif score >= 100 and percent >= 95:
        return "Good"
    elif score >= 100 and percent >= 90:
        return "Average"
    elif score < 100 and percent >= 90:
        return "Inexperienced"
    elif percent < 90:
        return "Low Rated"
    else:
        return "Uncategorized"

# Price analytics functions
def create_price_analytics(df):
    """Create price analytics dashboard"""
    if df.empty:
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = df['price'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col2:
        median_price = df['price'].median()
        st.metric("Median Price", f"${median_price:.2f}")
    
    with col3:
        deal_count = len(df[df['price'] < (avg_price * 0.85)])
        st.metric("Potential Deals", f"{deal_count} item(s)", 
                 help="Items priced 15% below average")
    
    with col4:
        total_listings = len(df)
        st.metric("Total Listings", f"{total_listings}")
    
    # Model breakdown if multiple models selected
    if 'model' in df.columns and df['model'].nunique() > 1:
        st.subheader("📊 Price by Model")
        model_stats = df.groupby('model').agg({
            'price': ['mean', 'median', 'count']
        }).round(2)
        model_stats.columns = ['Avg Price', 'Median Price', 'Count']
        model_stats = model_stats.reset_index()
        model_stats['Avg Price'] = model_stats['Avg Price'].apply(lambda x: f"${x:.2f}")
        model_stats['Median Price'] = model_stats['Median Price'].apply(lambda x: f"${x:.2f}")
        st.dataframe(model_stats, use_container_width=True)

    # Highlight best deals
    st.subheader("🎯 Best Deals (15% below average)")
    deals = df[df['price'] < (avg_price * 0.85)]
    if not deals.empty:
        # Convert back to formatted prices for display
        deals_display = df[df.index.isin(deals.index)].copy()
        deals_display['savings'] = deals_display.index.map(
            lambda x: f"${avg_price - df.loc[x, 'price']:.2f}"
        )
        display_cols = ['listing', 'condition', 'price', 'savings', 'seller', 'seller_rating', 'seller_feedback', 'link']
        if 'model' in deals_display.columns:
            display_cols.insert(1, 'model')
        
        st.dataframe(
            deals_display[display_cols],
            column_config={
                "link": st.column_config.LinkColumn("Link", display_text="View Deal"),
                "price": st.column_config.NumberColumn("price", format="$%.2f")
            },
            use_container_width=True
        )
    else:
        st.info("No significant deals found in current results.")

# Function to search for a specific headphone model
def search_headphone_model(model_name, category_id, listing_type_filter, seller_rating_filter, max_price, limit, access_token):
    """Search for a specific headphone model"""
    
    # Build query with model-specific exclusions
    if "Bose" in model_name or "Beats" in model_name or "Sony WH" in model_name:
        # For consumer headphones, exclude common accessories and parts
        query = f'"{model_name}" -(case,cover,cable,cord,charger,parts,broken,repair,box,manual,accessory,stand,holder)'
    else:
        # For professional headphones, be more lenient but still exclude obvious non-items
        query = f'"{model_name}" -(empty box,manual only,case only,for parts,broken,repair kit)'

    # Build filters with minimum price for non-auction listings
    if listing_type_filter in ["Fixed Price", "Best Offer"]:
        # Set minimum price to filter out accessories
        min_price = 25
        price_filter = f"price:[{min_price}..{max_price}]"
    else:
        # For auctions or "All" listings, use the original price filter
        price_filter = f"price:[1..{max_price}]"
    
    filters = [
        price_filter,
        "priceCurrency:USD",
        "conditions:{1000|1500|2000|2500|3000}"
    ]

    if listing_type_filter == "Auction":
        filters.append("buyingOptions:{AUCTION}")
    elif listing_type_filter == "Fixed Price":
        filters.append("buyingOptions:{FIXED_PRICE}")
    elif listing_type_filter == "Best Offer":
        filters.append("buyingOptions:{BEST_OFFER}")

    params = {
        "q": query,
        "filter": ",".join(filters),
        "limit": limit
    }

    if category_id:
        params["category_ids"] = category_id

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get("https://api.ebay.com/buy/browse/v1/item_summary/search", params=params, headers=headers)
    
    if response.status_code != 200:
        st.error(f"API Error for {model_name}: {response.status_code} - {response.text}")
        return []
    
    items = response.json().get("itemSummaries", [])
    results = []
    
    for item in items:
        title = item.get("title", "")
        price = float(item.get("price", {}).get("value", 0.0))
        shipping = float(item.get("shippingOptions", [{}])[0].get("shippingCost", {}).get("value", 0.0))
        total_cost = price + shipping
        link = item.get("itemWebUrl")
        buying_options = item.get("buyingOptions", [])
       
        # Filter out for parts not working (condition ID: 7000)
        condition_id = item.get("conditionId")
        if condition_id == "7000":
            continue

        # Get seller information
        seller_info = item.get("seller", {})
        seller_username = seller_info.get("username", "")
        seller_feedback_score = seller_info.get("feedbackScore", 0)
        seller_feedback_percent = seller_info.get("feedbackPercentage", 0)
        
        # Categorize seller
        seller_category = categorize_seller(seller_feedback_score, seller_feedback_percent)
        
        # Apply seller rating filter
        if seller_rating_filter and seller_category not in seller_rating_filter:
            continue

        end_time_str = item.get("itemEndDate")
        end_time = "N/A"
        if "AUCTION" in buying_options and end_time_str:
            try:
                utc_dt = datetime.datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                central = pytz.timezone('US/Central')
                local_dt = utc_dt.astimezone(central)
                end_time = local_dt.strftime("%Y-%m-%d %I:%M %p %Z")
            except Exception:
                end_time = "Invalid date"

        bid_count = item.get("bidCount") if "AUCTION" in buying_options else None

        if total_cost <= max_price:
            results.append({
                "listing": title,
                "condition": item.get("condition"),
                "price": price,
                "listing_type": ", ".join(buying_options),
                "model": model_name,
                "bid_count": bid_count,
                "auction_end_time": end_time,
                "seller": seller_username,
                "seller_rating": seller_category,
                "seller_feedback": seller_feedback_percent,
                "seller_feedback_score": seller_feedback_score,
                "link": link
            })
    
    return results

# UI
st.title("eBay Headphone Listings Search")
st.write("Search for specific headphone models across eBay with advanced filtering and analytics.")

# Main search interface
category_options = {
    "All Categories": None,
    "Headphones": "112529",
    "DJ & Monitoring Headphones": "14985",
}

# Headphone model options (A-Z order)
headphone_models = [
    "Audio Technica ATH-M50X",
    "Beats Solo3",
    "Beyerdynamic DT 770 Pro",
    "Bose QuietComfort 25",
    "Bose QuietComfort 35",
    "Sennheiser HD 280 Pro",
    "Sony MDR-7506",
    "Sony WH-1000XM4"
]

# Container approach for Select All
container = st.container()
all_models = st.checkbox("✅ Select All Models")

if all_models:
    selected_models = container.multiselect(
        "Select Headphone Model(s)",
        options=headphone_models,
        default=headphone_models,
        help="Select one or more headphone models to search for. Leave empty to use custom search term below."
    )
else:
    selected_models = container.multiselect(
        "Select Headphone Model(s)",
        options=headphone_models,
        help="Select one or more headphone models to search for. Leave empty to use custom search term below."
    )

# Custom search term (only used if no models selected)
custom_search_term = st.text_input(
    "Or enter custom search term:", 
    help="Only used if no headphone models are selected above"
)

selected_category = st.selectbox(
    "Category", 
    options=list(category_options.keys()),
    index=list(category_options.keys()).index('Headphones')
)

listing_type_filter = st.selectbox(
    "Filter by listing type",
    ["All", "Auction", "Fixed Price", "Best Offer"]
)

seller_rating_filter = st.multiselect(
    "Filter by seller rating (select multiple or leave empty for all)",
    ["Elite", "Excellent", "Very Good", "Good"],
    help=(
        """
        Elite: ≥5000/99% 
        Excellent: ≥1000/98% 
        Very Good: ≥500/97%
        Good: ≥100/95% 
        Average: ≥100/90% 
        Inexperienced: <100/≥90%
        Low Rated: <90%
    """
    )
)

max_price = st.number_input(
    "Maximum total price ($):", 
    min_value=1, 
    max_value=10000, 
    value=150
)

# Add information about minimum price filtering
if listing_type_filter in ["Fixed Price", "Best Offer"]:
    st.info("💡 For Fixed Price and Best Offer listings, a minimum price of $25 is automatically applied to filter out accessories and cables.")

limit = st.slider(
    "Number of listings per model:", 
    min_value=1, 
    max_value=100, 
    value=25,
    help="This is the limit per model, so selecting 3 models with limit 25 could return up to 75 results total"
)

# Search button
search_clicked = st.button("🔍 Search eBay", type="primary")

# Execute search
if search_clicked:
    if not access_token:
        st.error("Unable to search - missing access token")
    elif not selected_models and not custom_search_term:
        st.error("Please select at least one headphone model or enter a custom search term")
    else:
        # Determine what to search for
        search_items = selected_models if selected_models else [custom_search_term]
        category_id = category_options[selected_category]
        
        all_results = []
        
        # Search progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, search_item in enumerate(search_items):
            status_text.text(f"Searching for {search_item}...")
            progress_bar.progress((i + 1) / len(search_items))
            
            if selected_models:
                # Use headphone-specific search
                results = search_headphone_model(
                    search_item, category_id, listing_type_filter, 
                    seller_rating_filter, max_price, limit, access_token
                )
            else:
                # Use original search logic for custom terms
                results = search_headphone_model(
                    search_item, category_id, listing_type_filter,
                    seller_rating_filter, max_price, limit, access_token
                )
            
            all_results.extend(results)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Display results based on listing type
            if listing_type_filter != "Auction":
                df_sorted = df.sort_values(by="price").reset_index(drop=True)

                # Price Analytics Dashboard
                st.header("📊 Price Analytics")
                create_price_analytics(df_sorted)
                
                st.header("📋 Search Results")
                
                # Format currency columns
                def format_currency(val):
                    return f"${val:,.2f}"
                
                df_display = df_sorted.copy()
                for col in ["price"]:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].apply(format_currency)

                styled_df = df_display.style.set_properties(
                    **{"text-align": "center", "white-space": "pre-wrap"}
                ).set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold"), ("text-align", "center")]}
                ])

                st.dataframe(
                    styled_df,
                    column_config={
                        "link": st.column_config.LinkColumn("Link", display_text="View Listing")
                    },
                    use_container_width=True
                )
                
            else:  # Auction listings
                st.header("📋 Auction Listings")
                
                df_auctions = df.drop(columns=['price'] if 'price' in df.columns else [])
                df_auctions = df_auctions.sort_values(by="auction_end_time", ascending=True, na_position="last").reset_index(drop=True)

                styled_df = df_auctions.style.set_properties(
                    **{"text-align": "center", "white-space": "pre-wrap"}
                ).set_table_styles([
                    {"selector": "th", "props": [("font-weight", "bold"), ("text-align", "center")]}
                ])

                st.dataframe(
                    styled_df,
                    column_config={
                        "link": st.column_config.LinkColumn("Link", display_text="View Listing")
                    },
                    use_container_width=True
                )
            
            # Export functionality
            csv = df.to_csv(index=False)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"ebay_headphone_search_{timestamp}.csv"
            
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                filename,
                "text/csv"
            )
            
            # Summary
            models_searched = df['model'].nunique() if 'model' in df.columns else 1
            st.success(f"Found {len(all_results)} total listings across {models_searched} model(s)")
            
            # Show breakdown by model if multiple models
            if 'model' in df.columns and df['model'].nunique() > 1:
                st.subheader("📈 Results by Model")
                model_counts = df['model'].value_counts()
                for model, count in model_counts.items():
                    st.write(f"**{model}**: {count} listings")
        else:
            st.info("No listings found matching your criteria.")
