import pandas as pd
import matplotlib.pyplot as plt

start_year = 2015

# Read in the groundtruth (hospital data for USA)
hospital_df = pd.read_csv('./data/hospital_visits.csv')
hospital_df = hospital_df[['Year', 'Week', 'Total ILI', 'Total Patients']]

# Read in the virus dataframes
usa_df = pd.read_csv('./data/USA_flu_virus_counts.csv')
usa_df = usa_df[['Year', 'Week', 'ALL_INF']]
usa_df.rename(columns={'ALL_INF': 'VirusCount'}, inplace=True)
aus_df = pd.read_csv('./data/AUS_flu_virus_counts.csv')
aus_df = aus_df[['Year', 'Week', 'ALL_INF']]
aus_df.rename(columns={'ALL_INF': 'VirusCount'}, inplace=True)

# Join based on Year and Week
merged = pd.merge(left=aus_df, right=usa_df, on=['Year', 'Week'],
                  suffixes=['_AUS', '_USA'])
merged = pd.merge(left=hospital_df, right=merged, on=['Year', 'Week'])
merged = merged[merged.Year >= start_year]

# Plot the USA virus, AUS virus & USA hospitalization data together
x = [i for i in range(len(merged))]

fig, ax1 = plt.subplots()
color = 'tab:green'
ax1.set_xlabel("Week Number since {}".format(start_year))
ax1.set_ylabel('VirusCount', color=color)
# ax1.plot(merged['VirusCount_AUS'].tolist(), color=color)
# ax1.set(xlim=(0, 260), ylim=(0, 1500))
ax1.plot(merged['VirusCount_USA'].tolist(), color='tab:blue')
ax1.set(xlim=(0, 260))
ax1.tick_params(axis='y', labelcolor=color)
plt.yscale('log')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Total ILI', color=color)
ax2.plot(merged['Total ILI'].tolist(), color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(xlim=(0, 260))

plt.yscale('log')
fig.tight_layout()
plt.show()
