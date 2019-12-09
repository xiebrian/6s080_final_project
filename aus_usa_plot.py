
import pandas as pd
import matplotlib.pyplot as plt

start_year = 2012
week_offset = 26

# Read in dataframes and select important columns
aus_df = pd.read_csv('./data/AUS_flu_virus_counts.csv')
usa_df = pd.read_csv('./data/USA_flu_virus_counts.csv')
aus_df = aus_df[['Year', 'Week', 'ALL_INF']]
usa_df = usa_df[['Year', 'Week', 'ALL_INF']]
aus_df.rename(columns={'ALL_INF': 'Count'}, inplace=True)
usa_df.rename(columns={'ALL_INF': 'Count'}, inplace=True)

# Join based on Year and Week
merged = pd.merge(left=aus_df, right=usa_df, on=['Year', 'Week'],
                  suffixes=['_aus', '_usa'])
merged = merged[merged.Year >= start_year]

# 1. Plot the data together, without any offset
x = [i for i in range(len(merged))]
plt.plot(x, merged['Count_aus'].tolist())
plt.plot(x, merged['Count_usa'].tolist())
plt.yscale('log')
plt.xlabel("Weeks since {}".format(start_year))
plt.ylabel("Number of Flu Viruses")
plt.title("Number of Flu Viruses (AUS/USA) over Time")
plt.legend(labels=['AUS', 'USA'])
plt.show()

# 2. Plot the data together, offsetting AUS forward by 26 weeks
x_aus = [i for i in range(week_offset, len(merged) + week_offset)]
x_usa = [i for i in range(len(merged))]
plt.plot(x_aus, merged['Count_aus'].tolist())
plt.plot(x_usa, merged['Count_usa'].tolist())
plt.yscale('log')
plt.ylabel("Number of Flu Viruses")
plt.xlabel("Weeks since {} (AUS offset forwards {} weeks)".format(start_year, week_offset))
plt.title("Number of Flu Viruses (AUS/USA) over Time")
plt.legend(labels=['AUS', 'USA'])
plt.show()
