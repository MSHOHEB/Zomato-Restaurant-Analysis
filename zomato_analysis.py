"""
🍕 Zomato Restaurant Analysis — Python Data Analysis Project
=============================================================
Dataset  : 500 Indian Restaurants across 15 cities
Libraries: Pandas, Matplotlib, Seaborn, NumPy
Charts   : 8 Publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "zomato_restaurants.csv")
CHARTS = os.path.join(BASE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
PALETTE  = ["#E23744","#FF6B35","#F7C59F","#EFEFD0","#004E89","#1A936F","#88D498","#C6DABF","#F4F1DE","#3D405B"]
ZOMATO_R = "#E23744"
BG       = "#FAFAFA"
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

def save(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✅ {name}")

# ══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════
print("\n📦 Loading data...")
df = pd.read_csv(DATA)

print(f"  Shape      : {df.shape}")
print(f"  Nulls      : {df.isnull().sum().sum()}")
print(f"  Cities     : {df['city'].nunique()}")
print(f"  Avg Rating : {df['rating'].mean():.2f} ⭐")
print(f"  Price Range: ₹{df['avg_cost_for_two'].min()} — ₹{df['avg_cost_for_two'].max()}")

# Feature Engineering
df["rating_category"] = pd.cut(df["rating"],
    bins=[0,2.9,3.4,3.9,4.4,5.0],
    labels=["Poor(<3)","Average(3-3.4)","Good(3.5-3.9)","Very Good(4-4.4)","Excellent(4.5+)"])

df["cost_category"] = df["price_range"].str.extract(r'(\w+)')[0]

print("\n📊 Generating charts...")

# ══════════════════════════════════════════════════════════════
# 2. CHART 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("🍕 Zomato Restaurant Analysis — Overview Dashboard",
             fontsize=16, fontweight="bold", y=1.01, color=ZOMATO_R)

# 2a. City-wise Restaurant Count
city_count = df["city"].value_counts().head(12)
axes[0,0].bar(city_count.index, city_count.values, color=ZOMATO_R, edgecolor="white")
axes[0,0].set_title("Restaurants by City")
axes[0,0].set_ylabel("Count")
axes[0,0].tick_params(axis="x", rotation=45)
for bar in axes[0,0].patches:
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   str(int(bar.get_height())), ha="center", fontsize=7)

# 2b. Rating Distribution
axes[0,1].hist(df["rating"], bins=20, color=ZOMATO_R, edgecolor="white", linewidth=0.8)
axes[0,1].axvline(df["rating"].mean(), color="navy", linestyle="--",
                  label=f"Mean: {df['rating'].mean():.2f}⭐")
axes[0,1].axvline(df["rating"].median(), color="green", linestyle="--",
                  label=f"Median: {df['rating'].median():.2f}⭐")
axes[0,1].set_title("Rating Distribution")
axes[0,1].set_xlabel("Rating")
axes[0,1].legend()

# 2c. Online Order Pie
online = df["online_order"].value_counts()
axes[0,2].pie(online.values, labels=["Online Order\nAvailable","Not Available"],
              autopct="%1.1f%%", colors=[ZOMATO_R,"#CCCCCC"], startangle=140,
              wedgeprops={"edgecolor":"white","linewidth":2})
axes[0,2].set_title("Online Order Availability")

# 2d. Restaurant Type
rest_type = df["restaurant_type"].value_counts().head(8)
axes[1,0].barh(rest_type.index, rest_type.values, color=PALETTE[:8])
axes[1,0].set_title("Restaurant Type Distribution")
axes[1,0].set_xlabel("Count")

# 2e. Price Range
price_count = df["price_range"].value_counts()
short_labels = ["Budget","Affordable","Moderate","Premium","Luxury"]
axes[1,1].bar(short_labels[:len(price_count)], price_count.values,
              color=["#2ECC71","#F39C12","#E67E22","#E74C3C","#8E44AD"][:len(price_count)],
              edgecolor="white")
axes[1,1].set_title("Price Range Distribution")
axes[1,1].set_ylabel("Count")
axes[1,1].tick_params(axis="x", rotation=20)

# 2f. Table Booking
booking = df["table_booking"].value_counts()
axes[1,2].pie(booking.values, labels=["No Booking","Table Booking\nAvailable"],
              autopct="%1.1f%%", colors=["#CCCCCC",ZOMATO_R], startangle=140,
              wedgeprops={"edgecolor":"white","linewidth":2})
axes[1,2].set_title("Table Booking Availability")

plt.tight_layout()
save("01_overview_dashboard.png")

# ══════════════════════════════════════════════════════════════
# 3. CHART 2 — CITY ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("🏙️ City-wise Analysis", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Avg Rating by City
city_rating = df.groupby("city")["rating"].mean().sort_values(ascending=True)
axes[0].barh(city_rating.index, city_rating.values, color=ZOMATO_R)
axes[0].set_title("Avg Rating by City")
axes[0].set_xlabel("Avg Rating")
axes[0].axvline(df["rating"].mean(), color="navy", linestyle="--",
                label=f"Overall Avg: {df['rating'].mean():.2f}")
axes[0].legend()
for bar in axes[0].patches:
    axes[0].text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                 f"{bar.get_width():.2f}⭐", va="center", fontsize=9)

# Avg Cost by City
city_cost = df.groupby("city")["avg_cost_for_two"].mean().sort_values(ascending=False)
axes[1].bar(city_cost.index, city_cost.values, color=PALETTE[4], edgecolor="white")
axes[1].set_title("Avg Cost for Two by City (₹)")
axes[1].set_ylabel("Avg Cost (₹)")
axes[1].tick_params(axis="x", rotation=45)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 f"₹{int(bar.get_height())}", ha="center", fontsize=7, rotation=45)

plt.tight_layout()
save("02_city_analysis.png")

# ══════════════════════════════════════════════════════════════
# 4. CHART 3 — CUISINE ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("🍜 Cuisine Analysis", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Extract primary cuisine
df["primary_cuisine"] = df["cuisines"].str.split(",").str[0].str.strip()

# Top 15 cuisines by count
top_cuisines = df["primary_cuisine"].value_counts().head(15).sort_values()
axes[0].barh(top_cuisines.index, top_cuisines.values, color=ZOMATO_R)
axes[0].set_title("Top 15 Most Popular Cuisines")
axes[0].set_xlabel("Restaurant Count")

# Cuisine vs Avg Rating
cuisine_rating = df.groupby("primary_cuisine")["rating"].mean().sort_values(ascending=False).head(12)
axes[1].bar(cuisine_rating.index, cuisine_rating.values,
            color=PALETTE[:len(cuisine_rating)], edgecolor="white")
axes[1].set_title("Top Cuisines by Avg Rating")
axes[1].set_ylabel("Avg Rating")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_ylim(0, 5.5)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{bar.get_height():.2f}⭐", ha="center", fontsize=8)

plt.tight_layout()
save("03_cuisine_analysis.png")

# ══════════════════════════════════════════════════════════════
# 5. CHART 4 — RATING ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("⭐ Rating Deep Dive", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Rating Category Distribution
rating_cat = df["rating_category"].value_counts()
colors_rc = ["#E74C3C","#E67E22","#F39C12","#2ECC71","#27AE60"]
axes[0].bar(rating_cat.index, rating_cat.values, color=colors_rc[:len(rating_cat)], edgecolor="white")
axes[0].set_title("Rating Category Distribution")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 str(int(bar.get_height())), ha="center", fontsize=9)

# Rating vs Votes scatter
axes[1].scatter(df["votes"], df["rating"], alpha=0.5, color=ZOMATO_R, s=30)
axes[1].set_title("Rating vs Votes")
axes[1].set_xlabel("Number of Votes")
axes[1].set_ylabel("Rating")

# Online Order vs Rating
sns.boxplot(data=df, x="online_order", y="rating",
            palette=[ZOMATO_R,"#CCCCCC"], ax=axes[2])
axes[2].set_title("Rating: Online vs Offline")
axes[2].set_xlabel("Online Order Available")
axes[2].set_ylabel("Rating")

plt.tight_layout()
save("04_rating_analysis.png")

# ══════════════════════════════════════════════════════════════
# 6. CHART 5 — PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("💰 Price Range Analysis", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Price range vs Rating
price_order = ["Budget (₹0-200)","Affordable (₹200-500)","Moderate (₹500-1000)","Premium (₹1000-2000)","Luxury (₹2000+)"]
price_rating = df.groupby("price_range")["rating"].mean().reindex(price_order)
short = ["Budget","Affordable","Moderate","Premium","Luxury"]
axes[0].bar(short, price_rating.values,
            color=["#2ECC71","#F39C12","#E67E22","#E74C3C","#8E44AD"],
            edgecolor="white")
axes[0].set_title("Avg Rating by Price Range")
axes[0].set_ylabel("Avg Rating")
axes[0].set_ylim(0, 5.5)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{bar.get_height():.2f}⭐", ha="center", fontsize=9)

# Cost distribution by restaurant type
type_cost = df.groupby("restaurant_type")["avg_cost_for_two"].mean().sort_values(ascending=False).head(8)
axes[1].barh(type_cost.index, type_cost.values, color=PALETTE[4])
axes[1].set_title("Avg Cost by Restaurant Type (₹)")
axes[1].set_xlabel("Avg Cost for Two (₹)")

plt.tight_layout()
save("05_price_analysis.png")

# ══════════════════════════════════════════════════════════════
# 7. CHART 6 — TOP RESTAURANTS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("🏆 Top Restaurants Analysis", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Top 10 by Rating
top_rated = df.nlargest(10, "rating")[["name","city","rating","votes"]].sort_values("rating")
axes[0].barh(top_rated["name"] + " (" + top_rated["city"] + ")",
             top_rated["rating"], color=ZOMATO_R)
axes[0].set_title("Top 10 Highest Rated Restaurants")
axes[0].set_xlabel("Rating")
axes[0].set_xlim(4, 5.1)
for bar in axes[0].patches:
    axes[0].text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                 f"{bar.get_width():.1f}⭐", va="center", fontsize=9)

# Top 10 by Votes
top_voted = df.nlargest(10, "votes")[["name","city","votes","rating"]].sort_values("votes")
axes[1].barh(top_voted["name"] + " (" + top_voted["city"] + ")",
             top_voted["votes"], color=PALETTE[4])
axes[1].set_title("Top 10 Most Voted Restaurants")
axes[1].set_xlabel("Number of Votes")

plt.tight_layout()
save("06_top_restaurants.png")

# ══════════════════════════════════════════════════════════════
# 8. CHART 7 — FAMOUS DISHES & TRENDS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("🍽️ Famous Dishes & Trends", fontsize=15, fontweight="bold", color=ZOMATO_R)

# Famous dish distribution
dish_count = df["famous_dish"].value_counts().head(12).sort_values()
axes[0].barh(dish_count.index, dish_count.values, color=ZOMATO_R)
axes[0].set_title("Most Popular Famous Dishes")
axes[0].set_xlabel("Restaurant Count")

# Established Year trend
year_count = df["established_year"].value_counts().sort_index()
axes[1].plot(year_count.index, year_count.values, marker="o",
             color=ZOMATO_R, linewidth=2.5, markersize=6)
axes[1].fill_between(year_count.index, year_count.values, alpha=0.15, color=ZOMATO_R)
axes[1].set_title("Restaurants Opened by Year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("New Restaurants")

plt.tight_layout()
save("07_dishes_trends.png")

# ══════════════════════════════════════════════════════════════
# 9. CHART 8 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
num_cols = ["rating","votes","avg_cost_for_two","seating_capacity","established_year"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"shrink":0.8})
ax.set_title("Numerical Features — Correlation Heatmap",
             fontsize=14, fontweight="bold", color=ZOMATO_R)
plt.tight_layout()
save("08_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 10. KEY METRICS SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("📊 KEY METRICS SUMMARY")
print("="*55)
print(f"  Total Restaurants    : {len(df)}")
print(f"  Cities Covered       : {df['city'].nunique()}")
print(f"  Cuisines Available   : {df['primary_cuisine'].nunique()}")
print(f"  Avg Rating           : {df['rating'].mean():.2f} ⭐")
print(f"  Highest Rated        : {df.loc[df['rating'].idxmax(),'name']} ({df['rating'].max()}⭐)")
print(f"  Most Voted           : {df.loc[df['votes'].idxmax(),'name']} ({df['votes'].max()} votes)")
print(f"  Online Order %       : {(df['online_order']=='Yes').mean()*100:.1f}%")
print(f"  Table Booking %      : {(df['table_booking']=='Yes').mean()*100:.1f}%")
print(f"  Avg Cost for Two     : ₹{df['avg_cost_for_two'].mean():.0f}")
print(f"  Most Popular Cuisine : {df['primary_cuisine'].value_counts().idxmax()}")
print(f"  Top City (Count)     : {df['city'].value_counts().idxmax()}")
print(f"  Top City (Rating)    : {df.groupby('city')['rating'].mean().idxmax()}")
print("="*55)
print("\n✅ All 8 charts saved to /charts folder!")
