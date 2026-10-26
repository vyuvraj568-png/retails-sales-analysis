"""Generates a realistic SYNTHETIC retail sales dataset (seeded, reproducible)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 9000
cats = {"Electronics":(8000,45000,0.12),"Furniture":(5000,30000,0.08),
        "Clothing":(500,4000,0.30),"Grocery":(100,1500,0.10),"Sports":(800,9000,0.20)}
regions = {"North":0.28,"South":0.25,"East":0.20,"West":0.27}
segments = ["Consumer","Corporate","Home Office"]
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0,730,N), unit="D")
# seasonality: more orders in Oct-Dec (festive)
keep = np.array([rng.random() < (1.0 if d.month in (10,11,12) else 0.7) for d in dates])
dates = dates[keep]; n = len(dates)
cat = rng.choice(list(cats), n, p=[0.18,0.12,0.30,0.25,0.15])
lo = np.array([cats[c][0] for c in cat]); hi = np.array([cats[c][1] for c in cat])
price = rng.integers(lo, hi)
qty = rng.integers(1,6,n)
disc = rng.choice([0,0.05,0.1,0.2,0.3], n, p=[0.35,0.2,0.2,0.15,0.1])
sales = np.round(price*qty*(1-disc),2)
base_margin = np.array([cats[c][2] for c in cat])
profit = np.round(sales*(base_margin - disc*0.6 + rng.normal(0,0.03,n)),2)
df = pd.DataFrame({"order_id":[f"ORD{100000+i}" for i in range(n)],
  "order_date":dates.date,"customer_id":[f"C{rng.integers(1,1500):04d}" for _ in range(n)],
  "region":rng.choice(list(regions),n,p=list(regions.values())),
  "segment":rng.choice(segments,n,p=[0.55,0.30,0.15]),
  "category":cat,"unit_price":price,"quantity":qty,"discount":disc,"sales":sales,"profit":profit})
# add realistic dirt: missing values + duplicates
df.loc[rng.choice(n,60,replace=False),"region"]=np.nan
df.loc[rng.choice(n,40,replace=False),"segment"]=None
df = pd.concat([df, df.sample(25,random_state=1)]).sample(frac=1,random_state=2).reset_index(drop=True)
df.to_csv("data/retail_sales_raw.csv",index=False)
print("rows:",len(df))
