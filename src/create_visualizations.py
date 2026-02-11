"""
Create visualizations for SkyGeni Sales Intelligence Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data
df = pd.read_csv('data/skygeni_sales_data.csv')
df['created_date'] = pd.to_datetime(df['created_date'])
df['closed_date'] = pd.to_datetime(df['closed_date'])
df['closed_quarter'] = df['closed_date'].dt.to_period('Q')

print("Creating visualizations...")

# Create a figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Win Rate Trend Over Time
ax1 = plt.subplot(2, 3, 1)
quarterly_perf = df.groupby('closed_quarter').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum()
}).reset_index()
quarterly_perf['win_rate'] = quarterly_perf['outcome'] / quarterly_perf['deal_id'] * 100
quarters_str = quarterly_perf['closed_quarter'].astype(str)
ax1.plot(range(len(quarters_str)), quarterly_perf['win_rate'], marker='o', linewidth=2, markersize=8)
ax1.axhline(y=quarterly_perf['win_rate'].mean(), color='r', linestyle='--', alpha=0.7, label='Average')
ax1.set_xlabel('Quarter', fontsize=10)
ax1.set_ylabel('Win Rate (%)', fontsize=10)
ax1.set_title('Win Rate Trend Over Time', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(quarters_str)))
ax1.set_xticklabels(quarters_str, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Win Rate by Lead Source
ax2 = plt.subplot(2, 3, 2)
source_perf = df.groupby('lead_source').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum()
}).reset_index()
source_perf['win_rate'] = source_perf['outcome'] / source_perf['deal_id'] * 100
source_perf = source_perf.sort_values('win_rate', ascending=True)
colors = ['#ff6b6b' if x < 45 else '#51cf66' for x in source_perf['win_rate']]
ax2.barh(source_perf['lead_source'], source_perf['win_rate'], color=colors)
ax2.set_xlabel('Win Rate (%)', fontsize=10)
ax2.set_title('Win Rate by Lead Source', fontsize=12, fontweight='bold')
ax2.axvline(x=45, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(source_perf['win_rate']):
    ax2.text(v + 0.5, i, f'{v:.1f}%', va='center')

# 3. Sales Cycle Impact
ax3 = plt.subplot(2, 3, 3)
df['cycle_bucket'] = pd.cut(df['sales_cycle_days'], 
                             bins=[0, 30, 60, 90, 200], 
                             labels=['0-30', '31-60', '61-90', '90+'])
cycle_perf = df.groupby('cycle_bucket').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum()
}).reset_index()
cycle_perf['win_rate'] = cycle_perf['outcome'] / cycle_perf['deal_id'] * 100
ax3.bar(range(len(cycle_perf)), cycle_perf['win_rate'], color='#4ecdc4', edgecolor='black')
ax3.set_xlabel('Sales Cycle (days)', fontsize=10)
ax3.set_ylabel('Win Rate (%)', fontsize=10)
ax3.set_title('Win Rate by Sales Cycle Length', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(cycle_perf)))
ax3.set_xticklabels(cycle_perf['cycle_bucket'])
for i, v in enumerate(cycle_perf['win_rate']):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# 4. Industry Performance
ax4 = plt.subplot(2, 3, 4)
industry_perf = df.groupby('industry').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum(),
    'deal_amount': 'mean'
}).reset_index()
industry_perf['win_rate'] = industry_perf['outcome'] / industry_perf['deal_id'] * 100
industry_perf = industry_perf.sort_values('win_rate', ascending=False)
x = range(len(industry_perf))
width = 0.35
ax4_2 = ax4.twinx()
bars1 = ax4.bar([i - width/2 for i in x], industry_perf['win_rate'], width, label='Win Rate', color='#95e1d3')
bars2 = ax4_2.bar([i + width/2 for i in x], industry_perf['deal_amount'], width, label='Avg Deal Size', color='#f38181', alpha=0.7)
ax4.set_xlabel('Industry', fontsize=10)
ax4.set_ylabel('Win Rate (%)', fontsize=10, color='#95e1d3')
ax4_2.set_ylabel('Avg Deal Size ($)', fontsize=10, color='#f38181')
ax4.set_title('Win Rate & Deal Size by Industry', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(industry_perf['industry'], rotation=45, ha='right')
ax4.tick_params(axis='y', labelcolor='#95e1d3')
ax4_2.tick_params(axis='y', labelcolor='#f38181')
ax4.legend(loc='upper left')
ax4_2.legend(loc='upper right')

# 5. Deal Amount Distribution by Outcome
ax5 = plt.subplot(2, 3, 5)
won_deals = df[df['outcome'] == 'Won']['deal_amount']
lost_deals = df[df['outcome'] == 'Lost']['deal_amount']
ax5.hist([won_deals, lost_deals], bins=30, label=['Won', 'Lost'], color=['#51cf66', '#ff6b6b'], alpha=0.7)
ax5.set_xlabel('Deal Amount ($)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Deal Amount Distribution by Outcome', fontsize=12, fontweight='bold')
ax5.legend()
ax5.set_xlim(0, 100000)

# 6. Regional Performance
ax6 = plt.subplot(2, 3, 6)
region_perf = df.groupby('region').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum()
}).reset_index()
region_perf['win_rate'] = region_perf['outcome'] / region_perf['deal_id'] * 100
region_perf['loss_rate'] = 100 - region_perf['win_rate']
regions = region_perf['region']
won = region_perf['win_rate']
lost = region_perf['loss_rate']
x_pos = range(len(regions))
ax6.bar(x_pos, won, label='Win Rate', color='#51cf66')
ax6.bar(x_pos, lost, bottom=won, label='Loss Rate', color='#ff6b6b')
ax6.set_ylabel('Percentage (%)', fontsize=10)
ax6.set_title('Win/Loss Distribution by Region', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(regions, rotation=45, ha='right')
ax6.legend()
for i, (w, l) in enumerate(zip(won, lost)):
    ax6.text(i, w/2, f'{w:.1f}%', ha='center', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('outputs/sales_insights.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: outputs/sales_insights.png")

# Create additional focused visualizations
# Pipeline Health Dashboard
fig2, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(16, 12))

# 7. Lead Source Mix Change
df['is_recent'] = df['closed_quarter'].astype(str) >= '2024Q2'
recent_source = df[df['is_recent']].groupby('lead_source').size()
previous_source = df[~df['is_recent']].groupby('lead_source').size()

x = range(len(recent_source))
width = 0.35
ax7.bar([i - width/2 for i in x], previous_source.values, width, label='Previous Quarters', color='#a8dadc')
ax7.bar([i + width/2 for i in x], recent_source.values, width, label='Recent Quarters', color='#457b9d')
ax7.set_ylabel('Number of Deals', fontsize=10)
ax7.set_title('Lead Source Mix: Recent vs Previous Quarters', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(recent_source.index, rotation=45, ha='right')
ax7.legend()

# 8. Sales Cycle Trend
df_sorted = df.sort_values('closed_date')
df_sorted['rolling_avg_cycle'] = df_sorted['sales_cycle_days'].rolling(window=100).mean()
ax8.scatter(df_sorted['closed_date'], df_sorted['sales_cycle_days'], alpha=0.1, s=10, color='gray')
ax8.plot(df_sorted['closed_date'], df_sorted['rolling_avg_cycle'], color='#e63946', linewidth=2, label='100-deal Moving Average')
ax8.set_xlabel('Date', fontsize=10)
ax8.set_ylabel('Sales Cycle (days)', fontsize=10)
ax8.set_title('Sales Cycle Trend Over Time', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Product Type Performance
product_quarter = df.groupby(['closed_quarter', 'product_type']).agg({
    'outcome': lambda x: (x == 'Won').sum() / len(x) * 100
}).reset_index()
product_quarter.columns = ['quarter', 'product_type', 'win_rate']
quarters_unique = product_quarter['quarter'].unique()
for product in product_quarter['product_type'].unique():
    data = product_quarter[product_quarter['product_type'] == product]
    ax9.plot(range(len(data)), data['win_rate'], marker='o', label=product, linewidth=2)
ax9.set_xlabel('Quarter', fontsize=10)
ax9.set_ylabel('Win Rate (%)', fontsize=10)
ax9.set_title('Win Rate Trend by Product Type', fontsize=12, fontweight='bold')
ax9.set_xticks(range(len(quarters_unique)))
ax9.set_xticklabels([str(q) for q in quarters_unique], rotation=45)
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. Top vs Bottom Performers
# Load risk scores
risk_df = pd.read_csv('outputs/pipeline_risk_scores.csv')
rep_performance = df.groupby('sales_rep_id').agg({
    'deal_id': 'count',
    'outcome': lambda x: (x == 'Won').sum(),
    'deal_amount': 'sum'
}).reset_index()
rep_performance['win_rate'] = rep_performance['outcome'] / rep_performance['deal_id'] * 100
rep_performance = rep_performance.sort_values('win_rate', ascending=False)

top_5 = rep_performance.head(5)
bottom_5 = rep_performance.tail(5)

y_pos_top = range(len(top_5))
y_pos_bottom = range(len(bottom_5))

ax10_top = ax10
ax10_top.barh(y_pos_top, top_5['win_rate'], color='#06d6a0', alpha=0.8)
ax10_top.set_yticks(y_pos_top)
ax10_top.set_yticklabels(top_5['sales_rep_id'])
ax10_top.set_xlabel('Win Rate (%)', fontsize=10)
ax10_top.set_title('Top 5 vs Bottom 5 Sales Reps by Win Rate', fontsize=12, fontweight='bold')
ax10_top.invert_yaxis()
for i, v in enumerate(top_5['win_rate']):
    ax10_top.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')

# Add bottom 5 in red on the same chart
y_offset = len(top_5) + 1
y_pos_bottom_shifted = [y + y_offset for y in y_pos_bottom]
ax10_top.barh(y_pos_bottom_shifted, bottom_5['win_rate'], color='#ef476f', alpha=0.8)
current_yticks = list(y_pos_top)
current_labels = list(top_5['sales_rep_id'])
ax10_top.set_yticks(current_yticks + y_pos_bottom_shifted)
ax10_top.set_yticklabels(current_labels + list(bottom_5['sales_rep_id']))
for i, (v, y) in enumerate(zip(bottom_5['win_rate'], y_pos_bottom_shifted)):
    ax10_top.text(v + 1, y, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/sales_insights.png', dpi=300, bbox_inches='tight')
print("✓ Saved dashboard to: outputs/sales_insights.png")

print("\n✅ All visualizations created successfully!")
