# Ola Bike / Uber Ride Data Analysis
# Author: Vivek Saraswat
# Description: Clean, analyze, and visualize NCR ride bookings dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load Data
df = pd.read_csv("ncr_ride_bookings.csv")

# 2. Quick Overview
print("\n Dataset Info ")
print(df.info())
print("\n First 5 Rows ")
print(df.head())

# 3. Check Exact Column Names
print("\n Column Names ")
print(df.columns.tolist())

# Optional: Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# 4. Handle Missing Values
missing_values = df.isnull().sum()
print("\n Missing Values Before Cleaning ")
print(missing_values)

# Drop rows where critical fields are missing
critical_columns = [col for col in df.columns if "Request" in col or "Timestamp" in col or "Date" in col]
if critical_columns:
    df.dropna(subset=critical_columns, inplace=True)

# Identify columns for heatmap correlation
heatmap_numeric_cols = ['Avg VTAT', 'Avg CTAT', 'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating']

# Fill missing values in numeric columns relevant for heatmap with 0
for col in heatmap_numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0) # Use assignment instead of inplace=True

# Fill missing values in other numeric columns with 0 (if any)
for col in df.select_dtypes(include=np.number).columns:
    if col not in heatmap_numeric_cols:
        df[col] = df[col].fillna(0) # Use assignment instead of inplace=True


# Fill missing values in non-numeric columns with "Unknown"
for col in df.select_dtypes(exclude=np.number).columns:
    df[col] = df[col].fillna("Unknown") # Use assignment instead of inplace=True


print("\n Missing Values After Cleaning ")
print(df.isnull().sum())

# 5. Convert date/time columns
# Attempt to parse with a common format first, then fall back
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        try:
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
        except Exception:
            # If mixed format fails, try without specifying format
            try:
                 df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                 pass


# 6. Feature Engineering
if "Request Timestamp" in df.columns:
    df['Request Hour'] = df['Request Timestamp'].dt.hour
    df['Request Day'] = df['Request Timestamp'].dt.date
elif "Request timestamp" in df.columns:
    df['Request Hour'] = df['Request timestamp'].dt.hour
    df['Request Day'] = df['Request timestamp'].dt.date
elif "Date" in df.columns and "Time" in df.columns:
    try:
        # Combine Date and Time columns and parse with mixed format
        df['Request Timestamp'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), format='mixed', errors='coerce')
        df['Request Hour'] = df['Request Timestamp'].dt.hour
        df['Request Day'] = df['Request Timestamp'].dt.date
    except Exception as e:
        print(f"Could not create 'Request Timestamp' from 'Date' and 'Time': {e}")


# 7. Analysis & Insights

# Ride requests by hour
if "Request Hour" in df.columns:
    plt.figure(figsize=(10,6))
    sns.countplot(x="Request Hour", data=df, palette="viridis", legend=False)
    plt.title("Ride Requests by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Requests")
    plt.show()
    # Summarize insights for Ride Requests by Hour
    print("\n Key Insights - Ride Requests by Hour")
    peak_hour = df['Request Hour'].value_counts().idxmax()
    print(f"- Peak request hour: {peak_hour}:00 hrs")


# Ride status distribution
status_column = [col for col in df.columns if "status" in col.lower()]
if status_column:
    plt.figure(figsize=(8,5))
    sns.countplot(x=status_column[0], data=df, palette="coolwarm", legend=False)
    plt.title("Ride Status Distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.show()
    # Summarize insights for Ride Status Distribution
    print("\n Key Insights - Ride Status Distribution")
    most_common_status = df[status_column[0]].value_counts().idxmax()
    print(f"- Most common ride status: {most_common_status}")
    if 'Booking Status' in df.columns:
        status_counts = df['Booking Status'].value_counts()
        print("\n- Ride Status Distribution:")
        print(status_counts)


# Requests over time
if "Request Day" in df.columns:
    daily_requests = df.groupby("Request Day").size()
    plt.figure(figsize=(12,6))
    daily_requests.plot(marker="o")
    plt.title("Daily Ride Requests")
    plt.xlabel("Date")
    plt.ylabel("Number of Requests")
    plt.grid(True)
    plt.show()
    # Summarize insights for Requests over Time
    print("\n Key Insights - Daily Ride Requests")
    # Convert Request Day to datetime for proper sorting before finding the busiest day
    df['Request Day'] = pd.to_datetime(df['Request Day'], errors='coerce')
    df.dropna(subset=['Request Day'], inplace=True) # Drop rows where conversion failed
    if not df.empty:
        busiest_day = df['Request Day'].value_counts().idxmax().date() # Get the date part
        print(f"- Busiest day for bookings: {busiest_day}")
    else:
        print("- Could not determine busiest day as 'Request Day' column is empty after cleaning.")


# 8. Key Insights
print("\n Overall Key Insights ")


# 9. Save Cleaned Data
# -------------------- Data Analysis --------------------

# Rides per day
if 'Request Day' in df.columns:
    rides_per_day = df.groupby(df['Request Day']).size()
    plt.figure(figsize=(10,5))
    rides_per_day.plot()
    plt.title("Rides Per Day")
    plt.xlabel("Date")
    plt.ylabel("Number of Rides")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Summarize insights for Rides per Day
    print("\n Key Insights - Rides Per Day")
    if not rides_per_day.empty:
        busiest_day_overall = rides_per_day.idxmax().date()
        print(f"- Overall busiest day for bookings: {busiest_day_overall}")


vehicle_counts = df['Vehicle Type'].value_counts()

# Colors for better visuals
colors = plt.cm.Paired(range(len(vehicle_counts)))

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    vehicle_counts,
    labels=vehicle_counts.index,
    colors=colors,
    autopct='%1.1f%%',   # Show percentage
    startangle=90        # Start from the top
)

# Add title
plt.title("\n\n\nDistribution of Vehicle Types\n\n\n")
plt.axis('equal')  # Equal aspect ratio to make it circular
plt.show()

# Status distribution
if status_column:
    plt.figure(figsize=(6,4))
    df[status_column[0]].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Ride Status Distribution")
    plt.ylabel("Count")
    plt.show()
    # Summarize insights for Status distribution
    print("\n Key Insights - Ride Status Distribution (Bar Plot)")
    most_common_status_bar = df[status_column[0]].value_counts().idxmax()
    print(f"- Most common ride status: {most_common_status_bar}")


# Trip duration distribution
if 'trip_duration_min' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['trip_duration_min'], bins=30, kde=True, color='orange')
    plt.title("Trip Duration (Minutes)")
    plt.xlabel("Minutes")
    plt.show()
    # Summarize insights for Trip Duration Distribution
    print("\n Key Insights - Trip Duration Distribution")
    avg_trip_duration = df['trip_duration_min'].mean()
    print(f"- Average Trip Duration: {avg_trip_duration:.2f} minutes")


# Heatmap for correlations (numeric only)
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# Summarize insights from Heatmap (can be more detailed based on specific correlations)
print("\n Key Insights - Correlation Heatmap")





# Overall Summary of Key Insights
print("\n Overall Summary of Key Insights ")
if "Request Hour" in df.columns:
    peak_hour = df['Request Hour'].value_counts().idxmax()
    print(f"- Peak request hour: {peak_hour}:00 hrs")
if status_column:
    most_common_status = df[status_column[0]].value_counts().idxmax()
    print(f"- Most common ride status: {most_common_status}")
if "Request Day" in df.columns and not df['Request Day'].empty:
    busiest_day = df['Request Day'].value_counts().idxmax().date()
    print(f"- Busiest day for bookings: {busiest_day}")
if 'Pickup Location' in df.columns:
    most_common_pickup = df['Pickup Location'].value_counts().idxmax()
    print(f"- Most common pickup location: {most_common_pickup}")
if 'Booking Status' in df.columns:
    status_counts = df['Booking Status'].value_counts()
    print("\n- Ride Status Distribution:")
    print(status_counts)
if 'Ride Distance' in df.columns:
    avg_ride_distance = df['Ride Distance'].mean()
    print(f"\n- Average Ride Distance: {avg_ride_distance:.2f} units")
if 'Booking Value' in df.columns:
    avg_booking_value = df['Booking Value'].mean()
    print(f"- Average Booking Value: {avg_booking_value:.2f}")
if 'Vehicle Type' in df.columns:
    vehicle_type_counts = df['Vehicle Type'].value_counts()
    print("\n- Vehicle Type Distribution:")
    print(vehicle_type_counts)
if 'Payment Method' in df.columns:
    payment_method_counts = df['Payment Method'].value_counts()
    print("\n- Payment Method Distribution:")
    print(payment_method_counts)