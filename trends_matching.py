import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

trends = pd.read_csv('flu_google_trends.csv')
trends['Month'] = trends.Month.str.replace('-0', '-')
counts = pd.read_csv('FluViewPhase5Data/Weekly_Data_Counts_updated.csv')
counts['total'] = counts['INFLUENZA A']+counts['INFLUENZA B']+counts['INFLUENZA A AND B']+counts['UNKNOWN']
counts = counts[['WEEK_NUMBER', 'total']]
merged = trends.merge(counts, left_on = "Month", right_on = "WEEK_NUMBER").drop('WEEK_NUMBER', axis= 1)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (months since Jan 2004)')
ax1.set_ylabel('Google Trends volume of term "flu" in US (relative)', color=color)
ax1.plot(merged['flu: (United States)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Flu-associated hospitalizations in US', color=color)  # we already handled the x-label with ax1
ax2.plot(merged['total'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax1.set_title('US "flu" search volume and number of US flu-associated hospitalizations')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
