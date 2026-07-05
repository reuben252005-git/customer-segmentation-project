import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 600

def make_segment(n, age_mu, age_sd, income_mu, income_sd, freq_mu, freq_sd,
                  monetary_mu, monetary_sd, recency_mu, recency_sd, tenure_mu, tenure_sd):
    age = np.clip(rng.normal(age_mu, age_sd, n), 18, 75).round().astype(int)
    income = np.clip(rng.normal(income_mu, income_sd, n), 15000, 250000).round(-2)
    freq = np.clip(rng.normal(freq_mu, freq_sd, n), 1, 60).round().astype(int)
    monetary = np.clip(rng.normal(monetary_mu, monetary_sd, n), 20, 20000).round(2)
    recency = np.clip(rng.normal(recency_mu, recency_sd, n), 0, 365).round().astype(int)
    tenure = np.clip(rng.normal(tenure_mu, tenure_sd, n), 1, 120).round().astype(int)
    return age, income, freq, monetary, recency, tenure

# 4 underlying customer archetypes (unlabeled in the final data - clustering should rediscover them)
segments = []

# 1. High-value loyal frequent buyers
n1 = 150
a,i,f,m,r,t = make_segment(n1, 42, 9, 95000, 20000, 28, 8, 4200, 1500, 12, 8, 48, 20)
segments.append(pd.DataFrame({"Age":a,"AnnualIncome":i,"PurchaseFrequency":f,"TotalSpend":m,"RecencyDays":r,"TenureMonths":t}))

# 2. Young occasional low-spend shoppers
n2 = 180
a,i,f,m,r,t = make_segment(n2, 26, 5, 42000, 10000, 6, 3, 350, 180, 60, 30, 10, 6)
segments.append(pd.DataFrame({"Age":a,"AnnualIncome":i,"PurchaseFrequency":f,"TotalSpend":m,"RecencyDays":r,"TenureMonths":t}))

# 3. Middle-aged moderate spenders, price-sensitive, discount-driven
n3 = 170
a,i,f,m,r,t = make_segment(n3, 48, 10, 60000, 15000, 14, 5, 1100, 450, 35, 15, 30, 15)
segments.append(pd.DataFrame({"Age":a,"AnnualIncome":i,"PurchaseFrequency":f,"TotalSpend":m,"RecencyDays":r,"TenureMonths":t}))

# 4. At-risk / churned high past value but inactive recently
n4 = 100
a,i,f,m,r,t = make_segment(n4, 50, 12, 80000, 25000, 4, 2, 2500, 1200, 180, 60, 60, 25)
segments.append(pd.DataFrame({"Age":a,"AnnualIncome":i,"PurchaseFrequency":f,"TotalSpend":m,"RecencyDays":r,"TenureMonths":t}))

df = pd.concat(segments, ignore_index=True)

genders = rng.choice(["Female","Male","Other"], size=len(df), p=[0.49,0.48,0.03])
regions = rng.choice(["North","South","East","West","Central"], size=len(df))
channels = rng.choice(["Online","In-Store","Mobile App"], size=len(df), p=[0.5,0.3,0.2])
categories_pref = rng.choice(
    ["Electronics","Apparel","Home & Living","Beauty","Groceries","Sports"],
    size=len(df)
)
loyalty_member = rng.choice([1,0], size=len(df), p=[0.55,0.45])

df.insert(0, "CustomerID", [f"CUST_{i:04d}" for i in range(1, len(df)+1)])
df["Gender"] = genders
df["Region"] = regions
df["PreferredChannel"] = channels
df["PreferredCategory"] = categories_pref
df["LoyaltyMember"] = loyalty_member

df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("customer_data.csv", index=False)
print(df.shape)
print(df.head())
