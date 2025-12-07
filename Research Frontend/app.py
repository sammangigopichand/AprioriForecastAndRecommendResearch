# ====================== Flask App: Hybrid Apriori + Recommendations ======================
# Author: Gopi Chand
# Purpose: Generate Apriori frequent itemsets, association rules, recommendations, and visualizations
# Environment: Python 3.9+, Flask, Pandas, Mlxtend, Matplotlib

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import random
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ====================== 1. Helper Function: Load Data ======================
def load_data():
    df = pd.read_excel("online_retail_II.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.dropna(subset=['customer_id', 'description'], inplace=True)
    df['description'] = df['description'].str.strip().str.lower()
    return df

# ====================== 2. Generate Apriori Rules ======================
def generate_apriori_results():
    df = load_data()
    grouped = df.groupby(['customer_id', 'invoice'])['description'].apply(list)
    transactions = grouped.values.tolist()
    transactions = random.sample(transactions, min(3000, len(transactions)))

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_items = apriori(df_trans, min_support=0.004, use_colnames=True, max_len=2)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

    # Save charts
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.figure(figsize=(7, 4))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (Colored by Lift)')
    plt.tight_layout()
    plt.savefig("static/support_chart.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(rules['lift'], bins=30, color='teal', alpha=0.7)
    plt.title('Lift Value Distribution')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("static/lift_chart.png")
    plt.close()

    image_paths = ["support_chart.png", "lift_chart.png"]

    return frequent_items, rules, image_paths

# ====================== 3. Generate Recommendations ======================
def get_recommendations(product_name, rules):
    product_name = product_name.lower().strip()
    recs = []
    for _, row in rules.iterrows():
        if product_name in row['antecedents']:
            recs.extend(list(row['consequents']))
    recs = list(set(recs))
    return recs[:10] if recs else ["No strong recommendations found."]

# ====================== 4. Flask Routes ======================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    frequent_items, rules, image_paths = generate_apriori_results()
    rules = rules.head(5)  # ✅ Show only top 5 rules
    return render_template(
        'results.html',
        frequent_items=frequent_items,
        rules=rules,
        image_paths=image_paths,
        recommended_products=[]
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    product_name = request.form['product_name']
    frequent_items, rules, image_paths = generate_apriori_results()
    recommendations = get_recommendations(product_name, rules)
    rules = rules.head(5)  # ✅ Limit rules again here
    return render_template(
        'results.html',
        frequent_items=frequent_items,
        rules=rules,
        image_paths=image_paths,
        recommended_products=recommendations
    )


# ====================== 5. Run App ======================
if __name__ == '__main__':
    app.run(debug=True)
