import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Streamlit App Title
st.title('Liverpool Premier League Stats Analysis (2023/24 vs 2024/25)')

df1=pd.read_html('https://fbref.com/en/squads/822bd0ba/2023-2024/Liverpool-Stats', attrs={"id":"stats_standard_9"})[0]

#Clean first row
df1.columns = ['_'.join(col).strip() for col in df1.columns.values]

#Clean Column's Name
df1.columns = [col.replace('Unnamed: 0_level_0_', '') if 'Unnamed: 0_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 1_level_0_', '') if 'Unnamed: 1_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 2_level_0_', '') if 'Unnamed: 2_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 3_level_0_', '') if 'Unnamed: 3_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 4_level_0_', '') if 'Unnamed: 4_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 5_level_0_', '') if 'Unnamed: 5_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 6_level_0_', '') if 'Unnamed: 6_level_0_' in col else col for col in df1.columns]

# after column Pos, make all as numbers

# Find the index of the 'Pos' column
pos_col_index = df1.columns.get_loc('Pos')

# Iterate through columns after 'Pos' and convert to numeric, coercing errors
for col in df1.columns[pos_col_index + 1:]:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')

# Identify the name of the last column
last_column_name = df1.columns[-1]

# Drop the last column
df1 = df1.drop(columns=[last_column_name])

# Remove Letters on Nation column
df1['Nation'] = df1['Nation'].str[2:]

# Analyze Playing Time columns
playing_time_cols = ['MP','Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']
display(df1[playing_time_cols].describe())

# Display playing time per player
playing_time_per_player = df1[['Player','MP', 'Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']]
display(playing_time_per_player)

playing_time_per_player_cleaned = playing_time_per_player.dropna()
display(playing_time_per_player_cleaned)

playing_time_per_player_cleaned = playing_time_per_player_cleaned[:-2]
display(playing_time_per_player_cleaned)

# Create a new column combining Player name and MP
playing_time_per_player_cleaned['Player_MP'] = playing_time_per_player_cleaned['Player'] + ' (' + playing_time_per_player_cleaned['MP'].astype(str) + ' MP)'

plt.figure(figsize=(10, 8))
ax = sns.barplot(x='Playing Time_Min', y='Player_MP', data=playing_time_per_player_cleaned.sort_values('Playing Time_Min', ascending=False))
plt.title('Playing Time Min per Player (Games Played)')
plt.xlabel('Playing Time Min')
plt.ylabel('Player')
plt.tight_layout()

# Add the value of Playing Time_90s at the end of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height()/2.,
             '{:1.1f}'.format(width),
             ha='left', va='center')

plt.show()

# Calculate goals per minute played
# Avoid division by zero by adding a small value or filtering out players with 0 minutes
df1_cleaned = df1[df1['Playing Time_Min'] > 0].copy()
df1_cleaned['Goals_Per_Minute'] = df1_cleaned['Performance_Gls'] / df1_cleaned['Playing Time_Min']

# Display the relevant columns
display(df1_cleaned[['Player', 'Playing Time_Min', 'Performance_Gls', 'Goals_Per_Minute']].sort_values('Goals_Per_Minute', ascending=False))

# Display goals per 90 minutes played per player
goals_per_90 = df1[['Player', 'Per 90 Minutes_Gls']]

# Remove rows with NaN values in the 'Per 90 Minutes_Gls' column and the last two rows
goals_per_90_cleaned = goals_per_90.dropna(subset=['Per 90 Minutes_Gls'])
goals_per_90_cleaned = goals_per_90_cleaned[:-2]

# Remove players with 0.00 goals per 90 minutes
goals_per_90_cleaned_filtered = goals_per_90_cleaned[goals_per_90_cleaned['Per 90 Minutes_Gls'] > 0]

display(goals_per_90_cleaned_filtered.sort_values('Per 90 Minutes_Gls', ascending=False))

# Display assists per 90 minutes played per player
assists_per_90 = df1[['Player', 'Per 90 Minutes_Ast']]

# Remove rows with NaN values in the 'Per 90 Minutes_Ast' column and the last two rows
assists_per_90_cleaned = assists_per_90.dropna(subset=['Per 90 Minutes_Ast'])
assists_per_90_cleaned = assists_per_90_cleaned[:-2]

# Remove players with 0.00 assists per 90 minutes
assists_per_90_cleaned_filtered = assists_per_90_cleaned[assists_per_90_cleaned['Per 90 Minutes_Ast'] > 0]

display(assists_per_90_cleaned_filtered.sort_values('Per 90 Minutes_Ast', ascending=False))

# Display goals + assists per 90 minutes played per player
goals_assists_per_90 = df1[['Player', 'Per 90 Minutes_G+A']]

# Remove rows with NaN values in the 'Per 90 Minutes_G+A' column and the last two rows
goals_assists_per_90_cleaned = goals_assists_per_90.dropna(subset=['Per 90 Minutes_G+A'])
goals_assists_per_90_cleaned = goals_assists_per_90_cleaned[:-2]

# Remove players with 0.00 goals + assists per 90 minutes
goals_assists_per_90_cleaned_filtered = goals_assists_per_90_cleaned[goals_assists_per_90_cleaned['Per 90 Minutes_G+A'] > 0]

display(goals_assists_per_90_cleaned_filtered.sort_values('Per 90 Minutes_G+A', ascending=False))

# Select relevant columns for comparison
goals_comparison = df1[['Player', 'MP', 'Performance_Gls', 'Expected_xG']]

# Remove rows with NaN values in these columns
goals_comparison_cleaned = goals_comparison.dropna()

# Remove the last two rows (Squad Total and Opponent Total)
goals_comparison_cleaned = goals_comparison_cleaned[:-2]

# Filter for players with at least one goal
goals_comparison_cleaned = goals_comparison_cleaned[goals_comparison_cleaned['Performance_Gls'] > 0]

# Create a new column combining Player name and MP
goals_comparison_cleaned['Player_MP'] = goals_comparison_cleaned['Player'] + ' (' + goals_comparison_cleaned['MP'].astype(str) + ' MP)'

# Melt the DataFrame for easier plotting, only including the numeric goal columns
goals_melted = goals_comparison_cleaned.melt(id_vars=['Player_MP'], value_vars=['Performance_Gls', 'Expected_xG'], var_name='Metric', value_name='Value')


plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Value', y='Player_MP', hue='Metric', data=goals_melted.sort_values('Value', ascending=False))
plt.title('Comparison of Goals Scored and Expected Goals (2023/24 Season) - Players with Goals')
plt.xlabel('Count')
plt.ylabel('Player')
plt.legend(title='Metric')
plt.tight_layout()

# Add the value at the end of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height()/2.,
             '{:1.1f}'.format(width),
             ha='left', va='center')

plt.show()

# Select relevant columns for comparison
assists_comparison = df1[['Player', 'MP', 'Performance_Ast', 'Expected_xAG']]

# Remove rows with NaN values in these columns
assists_comparison_cleaned = assists_comparison.dropna()

# Remove the last two rows (Squad Total and Opponent Total)
assists_comparison_cleaned = assists_comparison_cleaned[:-2]

# Filter for players with at least one assist
assists_comparison_cleaned = assists_comparison_cleaned[assists_comparison_cleaned['Performance_Ast'] > 0]

# Create a new column combining Player name and MP
assists_comparison_cleaned['Player_MP'] = assists_comparison_cleaned['Player'] + ' (' + assists_comparison_cleaned['MP'].astype(str) + ' MP)'

# Melt the DataFrame for easier plotting, only including the numeric assist columns
assists_melted = assists_comparison_cleaned.melt(id_vars=['Player_MP'], value_vars=['Performance_Ast', 'Expected_xAG'], var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Value', y='Player_MP', hue='Metric', data=assists_melted.sort_values('Value', ascending=False))
plt.title('Comparison of Assists and Expected Assists (2023/24 Season) - Players with Assists')
plt.xlabel('Count')
plt.ylabel('Player')
plt.legend(title='Metric')
plt.tight_layout()

# Add the value at the end of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height()/2.,
             '{:1.1f}'.format(width),
             ha='left', va='center')

plt.show()

# Calculate non-penalty goals plus assists
df1['Performance_npxG+A'] = df1['Performance_G+A'] - df1['Performance_PK']

# Select relevant columns for comparison
npxg_assists_comparison = df1[['Player', 'MP', 'Performance_npxG+A', 'Expected_npxG+xAG']]

# Remove rows with NaN values in these columns
npxg_assists_comparison_cleaned = npxg_assists_comparison.dropna()

# Remove the last two rows (Squad Total and Opponent Total)
npxg_assists_comparison_cleaned = npxg_assists_comparison_cleaned[:-2]

# Filter for players with at least one non-penalty goal contribution
npxg_assists_comparison_cleaned = npxg_assists_comparison_cleaned[npxg_assists_comparison_cleaned['Performance_npxG+A'] > 0]

# Create a new column combining Player name and MP
npxg_assists_comparison_cleaned['Player_MP'] = npxg_assists_comparison_cleaned['Player'] + ' (' + npxg_assists_comparison_cleaned['MP'].astype(str) + ' MP)'

# Melt the DataFrame for easier plotting, only including the numeric columns
npxg_assists_melted = npxg_assists_comparison_cleaned.melt(id_vars=['Player_MP'], value_vars=['Performance_npxG+A', 'Expected_npxG+xAG'], var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Value', y='Player_MP', hue='Metric', data=npxg_assists_melted.sort_values('Value', ascending=False))
plt.title('Comparison of Non-Penalty Goals + Assists and Expected Non-Penalty Goals + Expected Assists (2023/24 Season)')
plt.xlabel('Count')
plt.ylabel('Player')
plt.legend(title='Metric')
plt.tight_layout()

# Add the value at the end of each bar
for p in ax.patches:
    width = p.get_width()
    plt.text(width, p.get_y() + p.get_height()/2.,
             '{:1.1f}'.format(width),
             ha='left', va='center')

plt.show()

from adjustText import adjust_text

plt.figure(figsize=(10, 8))

# Create a list of colors for the scatter plot points
point_colors = ['darkgreen' if row['Performance_npxG+A'] > row['Expected_npxG+xAG'] else 'black' for index, row in npxg_assists_comparison_cleaned.iterrows()]

ax = sns.scatterplot(x='Expected_npxG+xAG', y='Performance_npxG+A', data=npxg_assists_comparison_cleaned, c=point_colors)
plt.title('Scatter Plot of Non-Penalty Goals + Assists vs. Expected Non-Penalty Goals + Expected Assists (2023/24 Season)')
plt.xlabel('Expected Non-Penalty Goals + Expected Assists')
plt.ylabel('Non-Penalty Goals + Assists')

# Add a line for expected = actual
max_val = max(npxg_assists_comparison_cleaned['Expected_npxG+xAG'].max(), npxg_assists_comparison_cleaned['Performance_npxG+A'].max())
plt.plot([0, max_val], [0, max_val], 'k--', lw=1)


texts = []
for i, row in npxg_assists_comparison_cleaned.iterrows():
    # Keep text color as black or green based on previous logic
    color = 'green' if row['Performance_npxG+A'] > row['Expected_npxG+xAG'] else 'black'
    texts.append(plt.text(row['Expected_npxG+xAG'], row['Performance_npxG+A'], row['Player'], fontsize=9, color=color))

# Adjust text to avoid overlapping, trying to keep them closer
adjust_text(texts, force_points=(0.5, 0.5), force_text=(0.5, 0.5), arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))


plt.tight_layout()
plt.show()
