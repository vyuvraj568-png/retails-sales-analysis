"""Retail Sales Analysis: cleaning -> EDA -> insights. Run: python analysis.py"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")

# ---------- 1. LOAD & CLEAN ----------
df = pd.read_csv("data/retail_sales_raw.csv", parse_dates=["order_date"])
print("Raw shape:", df.shape)
dups = df.duplicated().sum(); df = df.drop_duplicates()
miss = df[["region","segment"]].isna().sum().to_dict()
df["region"] = df["region"].fillna("Unknown"); df["segment"] = df["segment"].fillna("Unknown")
df["year"]=df.order_date.dt.year; df["month"]=df.order_date.dt.to_period("M").astype(str)
df["month_num"]=df.order_date.dt.month
df["profit_margin"]=df.profit/df.sales
df.to_csv("data/retail_sales_clean.csv", index=False)
print(f"Removed duplicates: {dups} | Missing filled: {miss} | Clean shape: {df.shape}")

# ---------- 2. KPIs ----------
total_sales, total_profit = df.sales.sum(), df.profit.sum()
kpi = {"Total Sales":total_sales,"Total Profit":total_profit,
       "Overall Margin %":total_profit/total_sales*100,"Orders":df.order_id.nunique(),
       "Avg Order Value":df.sales.mean()}
for k,v in kpi.items(): print(f"{k}: {v:,.2f}")

# ---------- 3. ANALYSIS ----------
insights=[]
cat = df.groupby("category").agg(sales=("sales","sum"),profit=("profit","sum"),orders=("order_id","count"))
cat["margin_%"]=cat.profit/cat.sales*100; cat=cat.sort_values("sales",ascending=False)
reg = df.groupby("region").agg(sales=("sales","sum"),profit=("profit","sum")).sort_values("sales",ascending=False)
mon = df.groupby("month").sales.sum()
seg = df.groupby("segment").agg(sales=("sales","sum"),profit=("profit","sum"))
disc = df.groupby("discount").agg(avg_margin=("profit_margin","mean"),orders=("order_id","count"))
yr = df.groupby(["year","month_num"]).sales.sum().unstack(0)
q = df.groupby(df.order_date.dt.quarter).sales.sum()
top_cust = df.groupby("customer_id").sales.sum().sort_values(ascending=False)
top10_share = top_cust.head(int(len(top_cust)*0.1)).sum()/top_cust.sum()*100

insights.append(f"Top category by sales: {cat.index[0]} (Rs {cat.sales.iloc[0]:,.0f}), {cat.sales.iloc[0]/total_sales*100:.1f}% of total revenue.")
best_m = cat["margin_%"].idxmax(); worst_m = cat["margin_%"].idxmin()
insights.append(f"Highest margin category: {best_m} ({cat.loc[best_m,'margin_%']:.1f}%); lowest: {worst_m} ({cat.loc[worst_m,'margin_%']:.1f}%).")
insights.append(f"Top region: {reg.index[0]} with Rs {reg.sales.iloc[0]:,.0f} sales; weakest: {reg.drop('Unknown').index[-1]}.")
insights.append(f"Discounts hurt profitability: avg margin falls from {disc.avg_margin.loc[0]*100:.1f}% (no discount) to {disc.avg_margin.loc[0.3]*100:.1f}% (30% discount).")
loss = (df.profit<0).mean()*100
insights.append(f"{loss:.1f}% of orders are loss-making, mostly on heavy discounts.")
insights.append(f"Oct-Dec quarter (Q4) contributes {q.loc[4]/q.sum()*100:.1f}% of total sales: festive-season seasonality.")
insights.append(f"Top 10% customers generate {top10_share:.1f}% of revenue.")
open("insights.txt","w").write("\n".join(f"- {i}" for i in insights))
print("\nINSIGHTS\n"+"\n".join(insights))

# ---------- 4. CHARTS ----------
fig,ax=plt.subplots(figsize=(10,4.5)); mon.plot(ax=ax,marker="o",color="#2a6fdb")
ax.set_title("Monthly Sales Trend"); ax.set_ylabel("Sales (Rs)"); plt.xticks(rotation=60); plt.tight_layout(); plt.savefig("charts/01_monthly_sales_trend.png",dpi=130); plt.close()

fig,ax=plt.subplots(1,2,figsize=(12,4.5))
cat.sales.plot.bar(ax=ax[0],color="#2a6fdb"); ax[0].set_title("Sales by Category")
cat["margin_%"].plot.bar(ax=ax[1],color="#e07b39"); ax[1].set_title("Profit Margin % by Category")
plt.tight_layout(); plt.savefig("charts/02_category_sales_margin.png",dpi=130); plt.close()

fig,ax=plt.subplots(figsize=(7,4.5)); reg.sales.plot.bar(ax=ax,color="#3aa57a"); ax.set_title("Sales by Region")
plt.tight_layout(); plt.savefig("charts/03_region_sales.png",dpi=130); plt.close()

fig,ax=plt.subplots(figsize=(7,4.5)); (disc.avg_margin*100).plot(marker="o",ax=ax,color="#c0392b")
ax.set_title("Discount vs Avg Profit Margin"); ax.set_ylabel("Avg margin %"); ax.set_xlabel("Discount")
plt.tight_layout(); plt.savefig("charts/04_discount_vs_margin.png",dpi=130); plt.close()

pivot=df.pivot_table(index="category",columns="region",values="profit",aggfunc="sum")
fig,ax=plt.subplots(figsize=(8,4.5)); sns.heatmap(pivot,annot=True,fmt=".0f",cmap="YlGnBu",ax=ax)
ax.set_title("Profit Heatmap: Category x Region"); plt.tight_layout(); plt.savefig("charts/05_profit_heatmap.png",dpi=130); plt.close()
print("\nCharts saved in /charts")
