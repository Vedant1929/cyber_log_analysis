import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/login_logs.csv")

print("\nPreview Data:")
print(df.head())

print("\nColumn Info:")
print(df.info())

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
df.columns = df.columns.str.strip()

# -----------------------------
# CREATE TIMESTAMP
# -----------------------------
# Fixed: using 'DayOftheMonth' (lowercase 't')
df['timestamp'] = pd.to_datetime(
    df['Month'] + " " +
    df['DayOftheMonth'].astype(str) + " 2025 " +
    df['Time']
)

df['hour'] = df['timestamp'].dt.hour
df['day_name'] = df['timestamp'].dt.day_name()

# -----------------------------
# BASIC DATA STATS
# -----------------------------
print("\nTotal Records:", len(df))
print("Unique IPs:", df['IPAddress'].nunique())
print("Unique Usernames:", df['Username'].nunique())

# -----------------------------
# TOP ATTACKING IPs
# -----------------------------
top_ips = df['IPAddress'].value_counts().head(10)

plt.figure()
top_ips.plot(kind='bar')
plt.title("Top 10 IPs by Login Attempts")
plt.ylabel("Attempts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# MOST TARGETED USERNAMES
# -----------------------------
top_users = df['Username'].value_counts().head(10)

plt.figure()
sns.barplot(x=top_users.values, y=top_users.index)
plt.title("Top Targeted Usernames")
plt.xlabel("Attempts")
plt.show()

# -----------------------------
# ATTACKS BY HOUR
# -----------------------------
hourly = df['hour'].value_counts().sort_index()

plt.figure()
plt.plot(hourly.index, hourly.values, marker='o')
plt.title("Login Attempts by Hour")
plt.xlabel("Hour")
plt.ylabel("Attempts")
plt.show()

# -----------------------------
# ATTACK HEATMAP (DAY vs HOUR)
# -----------------------------
heatmap_data = df.pivot_table(
    index='day_name',
    columns='hour',
    values='IPAddress',
    aggfunc='count'
)

plt.figure(figsize=(10,5))
sns.heatmap(heatmap_data)
plt.title("Attack Heatmap (Day vs Hour)")
plt.show()

# -----------------------------
# PORT TARGET ANALYSIS
# -----------------------------
top_ports = df['Port'].value_counts().head(10)

plt.figure()
top_ports.plot(kind='bar')
plt.title("Most Targeted Ports")
plt.ylabel("Attempts")
plt.show()

# -----------------------------
# RISK SCORING PER IP
# -----------------------------
df['ip_attempts'] = df.groupby('IPAddress')['IPAddress'].transform('count')

# brute force threshold
threshold = 20
suspicious_ips = df[df['ip_attempts'] > threshold]['IPAddress'].unique()

print("\nSuspicious IPs (>20 attempts):")
print(suspicious_ips)

# -----------------------------
# HIGH RISK RECORDS PREVIEW
# -----------------------------
high_risk_records = df[df['IPAddress'].isin(suspicious_ips)]

print("\nSample High Risk Records:")
print(high_risk_records.head())

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number)

plt.figure()
sns.heatmap(numeric_cols.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

print("\nAnalysis Completed ✅")