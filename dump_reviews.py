import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
with open("reviews_for_labeling.txt", "w", encoding="utf-8") as f:
    for i, r in df.iterrows():
        f.write(f"[{i}] ({r['Rating']}, {r['Date']}) {r['Title']}: {r['Review']}\n\n")