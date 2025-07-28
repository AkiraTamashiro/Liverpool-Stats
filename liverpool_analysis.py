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




if not df1.empty:
    # --- 2023/24 Season Analysis ---
    st.header('2023/24 Season Analysis')

    st.subheader('Playing Time Analysis')
    playing_time_cols_df1 = ['MP','Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']
    if all(col in df1.columns for col in playing_time_cols_df1):
        st.dataframe(df1[playing_time_cols_df1].describe())

        playing_time_per_player_df1 = df1[['Player','MP', 'Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']].dropna().iloc[:-2].copy()
        playing_time_per_player_cleaned_df1 = playing_time_per_player_df1.copy() # Already cleaned above
        playing_time_per_player_cleaned_df1['Player_MP'] = playing_time_per_player_cleaned_df1['Player'] + ' (' + playing_time_per_player_cleaned_df1['MP'].astype(str) + ' MP)'

        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Playing Time_Min', y='Player_MP', hue=None, data=playing_time_per_player_cleaned_df1.sort_values('Playing Time_Min', ascending=False), ax=ax1) # Removed hue for single season
        ax1.set_title('Playing Time Min per Player (Games Played) - 2023/24 Season')
        ax1.set_xlabel('Playing Time Min')
        ax1.set_ylabel('Player')
        plt.tight_layout()
        for p in ax1.patches:
            width = p.get_width()
            ax1.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.write("Playing Time columns not found in 2023/24 data.")


    st.subheader('Goals Analysis')
    if 'Playing Time_Min' in df1.columns and 'Performance_Gls' in df1.columns:
        df1_cleaned_goals = df1[df1['Playing Time_Min'] > 0].copy()
        df1_cleaned_goals['Goals_Per_Minute'] = df1_cleaned_goals['Performance_Gls'] / df1_cleaned_goals['Playing Time_Min']
        st.dataframe(df1_cleaned_goals[['Player', 'Playing Time_Min', 'Performance_Gls', 'Goals_Per_Minute']].sort_values('Goals_Per_Minute', ascending=False))
    else:
        st.write("Required columns for Goals per Minute not found in 2023/24 data.")

    if 'Per 90 Minutes_Gls' in df1.columns:
        goals_per_90_df1 = df1[['Player', 'Per 90 Minutes_Gls']].dropna(subset=['Per 90 Minutes_Gls']).iloc[:-2].copy()
        goals_per_90_cleaned_filtered_df1 = goals_per_90_df1[goals_per_90_df1['Per 90 Minutes_Gls'] > 0].copy()
        st.dataframe(goals_per_90_cleaned_filtered_df1.sort_values('Per 90 Minutes_Gls', ascending=False))
    else:
        st.write("Per 90 Minutes Goals column not found in 2023/24 data.")

    if all(col in df1.columns for col in ['Player', 'MP', 'Performance_Gls', 'Expected_xG']):
        goals_comparison_df1 = df1[['Player', 'MP', 'Performance_Gls', 'Expected_xG']].dropna().iloc[:-2].copy()
        goals_comparison_cleaned_df1 = goals_comparison_df1[goals_comparison_df1['Performance_Gls'] > 0].copy()
        goals_comparison_cleaned_df1['Player_MP'] = goals_comparison_cleaned_df1['Player'] + ' (' + goals_comparison_cleaned_df1['MP'].astype(str) + ' MP)'
        goals_melted_df1 = goals_comparison_cleaned_df1.melt(id_vars=['Player_MP'], value_vars=['Performance_Gls', 'Expected_xG'], var_name='Metric', value_name='Value')

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=goals_melted_df1.sort_values('Value', ascending=False), ax=ax2)
        ax2.set_title('Comparison of Goals Scored and Expected Goals (2023/24 Season) - Players with Goals')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Player')
        ax2.legend(title='Metric')
        plt.tight_layout()
        for p in ax2.patches:
            width = p.get_width()
            ax2.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig2)
        plt.close(fig2)
    else:
         st.write("Required columns for Goals vs Expected Goals comparison not found in 2023/24 data.")


    st.subheader('Assists Analysis')
    if 'Per 90 Minutes_Ast' in df1.columns:
        assists_per_90_df1 = df1[['Player', 'Per 90 Minutes_Ast']].dropna(subset=['Per 90 Minutes_Ast']).iloc[:-2].copy()
        assists_per_90_cleaned_filtered_df1 = assists_per_90_df1[assists_per_90_df1['Per 90 Minutes_Ast'] > 0].copy()
        st.dataframe(assists_per_90_cleaned_filtered_df1.sort_values('Per 90 Minutes_Ast', ascending=False))
    else:
        st.write("Per 90 Minutes Assists column not found in 2023/24 data.")

    if all(col in df1.columns for col in ['Player', 'MP', 'Performance_Ast', 'Expected_xAG']):
        assists_comparison_df1 = df1[['Player', 'MP', 'Performance_Ast', 'Expected_xAG']].dropna().iloc[:-2].copy()
        assists_comparison_cleaned_df1 = assists_comparison_df1[assists_comparison_df1['Performance_Ast'] > 0].copy()
        assists_comparison_cleaned_df1['Player_MP'] = assists_comparison_cleaned_df1['Player'] + ' (' + assists_comparison_cleaned_df1['MP'].astype(str) + ' MP)'
        assists_melted_df1 = assists_comparison_cleaned_df1.melt(id_vars=['Player_MP'], value_vars=['Performance_Ast', 'Expected_xAG'], var_name='Metric', value_name='Value')

        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=assists_melted_df1.sort_values('Value', ascending=False), ax=ax3)
        ax3.set_title('Comparison of Assists and Expected Assists (2023/24 Season) - Players with Assists')
        ax3.set_xlabel('Count')
        ax3.set_ylabel('Player')
        ax3.legend(title='Metric')
        plt.tight_layout()
        for p in ax3.patches:
            width = p.get_width()
            ax3.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.write("Required columns for Assists vs Expected Assists comparison not found in 2023/24 data.")


    st.subheader('Non-Penalty Goals + Assists Analysis')
    if all(col in df1.columns for col in ['Player', 'MP', 'Performance_npxG+A', 'Expected_npxG+xAG']):
        npxg_assists_comparison_df1 = df1[['Player', 'MP', 'Performance_npxG+A', 'Expected_npxG+xAG']].dropna().iloc[:-2].copy()
        npxg_assists_comparison_cleaned_df1 = npxg_assists_comparison_df1[npxg_assists_comparison_df1['Performance_npxG+A'] > 0].copy()
        npxg_assists_comparison_cleaned_df1['Player_MP'] = npxg_assists_comparison_cleaned_df1['Player'] + ' (' + npxg_assists_comparison_cleaned_df1['MP'].astype(str) + ' MP)'
        npxg_assists_melted_df1 = npxg_assists_comparison_cleaned_df1.melt(id_vars=['Player_MP'], value_vars=['Performance_npxG+A', 'Expected_npxG+xAG'], var_name='Metric', value_name='Value')

        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=npxg_assists_melted_df1.sort_values('Value', ascending=False), ax=ax4)
        ax4.set_title('Comparison of Non-Penalty Goals + Assists and Expected Non-Penalty Goals + Expected Assists (2023/24 Season)')
        ax4.set_xlabel('Count')
        ax4.set_ylabel('Player')
        ax4.legend(title='Metric')
        plt.tight_layout()
        for p in ax4.patches:
            width = p.get_width()
            ax4.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig4)
        plt.close(fig4)

        fig5, ax5 = plt.subplots(figsize=(10, 8))
        point_colors_df1 = ['darkgreen' if row['Performance_npxG+A'] > row['Expected_npxG+xAG'] else 'black' for index, row in npxg_assists_comparison_cleaned_df1.iterrows()]
        ax5.scatter(x='Expected_npxG+xAG', y='Performance_npxG+A', data=npxg_assists_comparison_cleaned_df1, c=point_colors_df1)
        ax5.set_title('Scatter Plot of Non-Penalty Goals + Assists vs. Expected Non-Penalty Goals + Expected Assists (2023/24 Season)')
        ax5.set_xlabel('Expected Non-Penalty Goals + Expected Assists')
        ax5.set_ylabel('Non-Penalty Goals + Assists')
        max_val_df1 = max(npxg_assists_comparison_cleaned_df1['Expected_npxG+xAG'].max(), npxg_assists_comparison_cleaned_df1['Performance_npxG+A'].max())
        ax5.plot([0, max_val_df1], [0, max_val_df1], 'k--', lw=1)
        texts_df1 = []
        for i, row in npxg_assists_comparison_cleaned_df1.iterrows():
            color = 'green' if row['Performance_npxG+A'] > row['Expected_npxG+xAG'] else 'black'
            texts_df1.append(ax5.text(row['Expected_npxG+xAG'], row['Performance_npxG+A'], row['Player'], fontsize=9, color=color))
        adjust_text(texts_df1, force_points=(0.1, 0.1), force_text=(0.1, 0.1), arrowprops=dict(arrowstyle='-', color='grey', lw=0.8))
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)
    else:
        st.write("Required columns for Non-Penalty Goals + Assists analysis not found in 2023/24 data.")


if not df2.empty:
    # --- 2024/25 Season Analysis ---
    st.header('2024/25 Season Analysis')

    st.subheader('Playing Time Analysis')
    playing_time_cols_df2 = ['MP','Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']
    if all(col in df2.columns for col in playing_time_cols_df2):
        st.dataframe(df2[playing_time_cols_df2].describe())

        playing_time_per_player_df2 = df2[['Player','MP', 'Playing Time_Starts', 'Playing Time_Min', 'Playing Time_90s']].dropna().iloc[:-2].copy()
        playing_time_per_player_cleaned_df2 = playing_time_per_player_df2.copy() # Already cleaned above
        playing_time_per_player_cleaned_df2['Player_MP'] = playing_time_per_player_cleaned_df2['Player'] + ' (' + playing_time_per_player_cleaned_df2['MP'].astype(str) + ' MP)'

        fig6, ax6 = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Playing Time_Min', y='Player_MP', hue=None, data=playing_time_per_player_cleaned_df2.sort_values('Playing Time_Min', ascending=False), ax=ax6) # Removed hue for single season
        ax6.set_title('Playing Time Min per Player (Games Played) - 2024/25 Season')
        ax6.set_xlabel('Playing Time Min')
        ax6.set_ylabel('Player')
        plt.tight_layout()
        for p in ax6.patches:
            width = p.get_width()
            ax6.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig6)
        plt.close(fig6)
    else:
        st.write("Playing Time columns not found in 2024/25 data.")


    st.subheader('Goals Analysis')
    if 'Playing Time_Min' in df2.columns and 'Performance_Gls' in df2.columns:
        df2_cleaned_goals = df2[df2['Playing Time_Min'] > 0].copy()
        df2_cleaned_goals['Goals_Per_Minute'] = df2_cleaned_goals['Performance_Gls'] / df2_cleaned_goals['Playing Time_Min']
        st.dataframe(df2_cleaned_goals[['Player', 'Playing Time_Min', 'Performance_Gls', 'Goals_Per_Minute']].sort_values('Goals_Per_Minute', ascending=False))
    else:
        st.write("Required columns for Goals per Minute not found in 2024/25 data.")

    if 'Per 90 Minutes_Gls' in df2.columns:
        goals_per_90_df2 = df2[['Player', 'Per 90 Minutes_Gls']].dropna(subset=['Per 90 Minutes_Gls']).iloc[:-2].copy()
        goals_per_90_cleaned_filtered_df2 = goals_per_90_df2[goals_per_90_df2['Per 90 Minutes_Gls'] > 0].copy()
        st.dataframe(goals_per_90_cleaned_filtered_df2.sort_values('Per 90 Minutes_Gls', ascending=False))
    else:
        st.write("Per 90 Minutes Goals column not found in 2024/25 data.")

    # Using corrected column name for df2
    if all(col in df2.columns for col in ['Player', 'MP', 'Performance_Gls', 'Expected_xG']):
        goals_comparison_df2 = df2[['Player', 'MP', 'Performance_Gls', 'Expected_xG']].dropna().iloc[:-2].copy() # Use Expected_xG for df2
        goals_comparison_cleaned_df2 = goals_comparison_df2[goals_comparison_df2['Performance_Gls'] > 0].copy()
        goals_comparison_cleaned_df2['Player_MP'] = goals_comparison_cleaned_df2['Player'] + ' (' + goals_comparison_cleaned_df2['MP'].astype(str) + ' MP)'
        goals_melted_df2 = goals_comparison_cleaned_df2.melt(id_vars=['Player_MP'], value_vars=['Performance_Gls', 'Expected_xG'], var_name='Metric', value_name='Value') # Use Expected_xG for df2

        fig7, ax7 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=goals_melted_df2.sort_values('Value', ascending=False), ax=ax7)
        ax7.set_title('Comparison of Goals Scored and Expected Goals (2024/25 Season) - Players with Goals')
        ax7.set_xlabel('Count')
        ax7.set_ylabel('Player')
        ax7.legend(title='Metric')
        plt.tight_layout()
        for p in ax7.patches:
            width = p.get_width()
            ax7.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig7)
        plt.close(fig7)
    else:
         st.write("Required columns for Goals vs Expected Goals comparison not found in 2024/25 data.")


    st.subheader('Assists Analysis')
    if 'Per 90 Minutes_Ast' in df2.columns:
        assists_per_90_df2 = df2[['Player', 'Per 90 Minutes_Ast']].dropna(subset=['Per 90 Minutes_Ast']).iloc[:-2].copy()
        assists_per_90_cleaned_filtered_df2 = assists_per_90_df2[assists_per_90_df2['Per 90 Minutes_Ast'] > 0].copy()
        st.dataframe(assists_per_90_cleaned_filtered_df2.sort_values('Per 90 Minutes_Ast', ascending=False))
    else:
        st.write("Per 90 Minutes Assists column not found in 2024/25 data.")

    # Using corrected column name for df2
    if all(col in df2.columns for col in ['Player', 'MP', 'Performance_Ast', 'Expected_xAG']):
        assists_comparison_df2 = df2[['Player', 'MP', 'Performance_Ast', 'Expected_xAG']].dropna().iloc[:-2].copy() # Use Expected_xAG for df2
        assists_comparison_cleaned_df2 = assists_comparison_df2[assists_comparison_df2['Performance_Ast'] > 0].copy()
        assists_comparison_cleaned_df2['Player_MP'] = assists_comparison_cleaned_df2['Player'] + ' (' + assists_comparison_cleaned_df2['MP'].astype(str) + ' MP)'
        assists_melted_df2 = assists_comparison_cleaned_df2.melt(id_vars=['Player_MP'], value_vars=['Performance_Ast', 'Expected_xAG'], var_name='Metric', value_name='Value') # Use Expected_xAG for df2

        fig8, ax8 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=assists_melted_df2.sort_values('Value', ascending=False), ax=ax8)
        ax8.set_title('Comparison of Assists and Expected Assists (2024/25 Season) - Players with Assists')
        ax8.set_xlabel('Count')
        ax8.set_ylabel('Player')
        ax8.legend(title='Metric')
        plt.tight_layout()
        for p in ax8.patches:
            width = p.get_width()
            ax8.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig8)
        plt.close(fig8)
    else:
        st.write("Required columns for Assists vs Expected Assists comparison not found in 2024/25 data.")


    st.subheader('Non-Penalty Goals + Assists Analysis')
    # Using corrected column names for df2
    if all(col in df2.columns for col in ['Player', 'MP', 'Performance_npxG+A', 'Expected_npxG_xAG']):
        npxg_assists_comparison_df2 = df2[['Player', 'MP', 'Performance_npxG+A', 'Expected_npxG_xAG']].dropna().iloc[:-2].copy() # Use Expected_npxG_xAG for df2
        npxg_assists_comparison_cleaned_df2 = npxg_assists_comparison_df2[npxg_assists_comparison_df2['Performance_npxG+A'] > 0].copy()
        npxg_assists_comparison_cleaned_df2['Player_MP'] = npxg_assists_comparison_cleaned_df2['Player'] + ' (' + npxg_assists_comparison_cleaned_df2['MP'].astype(str) + ' MP)'
        npxg_assists_melted_df2 = npxg_assists_comparison_cleaned_df2.melt(id_vars=['Player_MP'], value_vars=['Performance_npxG+A', 'Expected_npxG_xAG'], var_name='Metric', value_name='Value') # Use Expected_npxG_xAG for df2

        fig9, ax9 = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Value', y='Player_MP', hue='Metric', data=npxg_assists_melted_df2.sort_values('Value', ascending=False), ax=ax9)
        ax9.set_title('Comparison of Non-Penalty Goals + Assists and Expected Non-Penalty Goals + Expected Assists (2024/25 Season)')
        ax9.set_xlabel('Count')
        ax9.set_ylabel('Player')
        ax9.legend(title='Metric')
        plt.tight_layout()
        for p in ax9.patches:
            width = p.get_width()
            ax9.text(width, p.get_y() + p.get_height()/2.,
                     '{:1.1f}'.format(width),
                     ha='left', va='center')
        st.pyplot(fig9)
        plt.close(fig9)

        fig10, ax10 = plt.subplots(figsize=(10, 8))
        point_colors_df2 = ['darkgreen' if row['Performance_npxG+A'] > row['Expected_npxG_xAG'] else 'black' for index, row in npxg_assists_comparison_cleaned_df2.iterrows()] # Use Expected_npxG_xAG for df2
        ax10.scatter(x='Expected_npxG_xAG', y='Performance_npxG+A', data=npxg_assists_comparison_cleaned_df2, c=point_colors_df2) # Use Expected_npxG_xAG for df2
        ax10.set_title('Scatter Plot of Non-Penalty Goals + Assists vs. Expected Non-Penalty Goals + Expected Assists (2024/25 Season)')
        ax10.set_xlabel('Expected Non-Penalty Goals + Expected Assists')
        ax10.set_ylabel('Non-Penalty Goals + Assists')
        max_val_df2 = max(npxg_assists_comparison_cleaned_df2['Expected_npxG_xAG'].max(), npxg_assists_comparison_cleaned_df2['Performance_npxG+A'].max()) # Use Expected_npxG_xAG for df2
        ax10.plot([0, max_val_df2], [0, max_val_df2], 'k--', lw=1)
        texts_df2 = []
        for i, row in npxg_assists_comparison_cleaned_df2.iterrows():
            color = 'green' if row['Performance_npxG+A'] > row['Expected_npxG_xAG'] else 'black' # Use Expected_npxG_xAG for df2
            texts_df2.append(ax10.text(row['Expected_npxG_xAG'], row['Performance_npxG+A'], row['Player'], fontsize=9, color=color)) # Corrected x and y back to original columns
        adjust_text(texts_df2, force_points=(0.1, 0.1), force_text=(0.1, 0.1), arrowprops=dict(arrowstyle='-', color='grey', lw=0.8))
        plt.tight_layout()
        st.pyplot(fig10)
        plt.close(fig10)

    else:
        st.write("Required columns for Non-Penalty Goals + Assists analysis not found in 2024/25 data.")


# --- Seasonal Comparison ---
st.header('Seasonal Comparison (2023/24 vs 2024/25)')

st.subheader('Goals Comparison')
if all(col in df1.columns for col in ['Player', 'Performance_Gls']) and all(col in df2.columns for col in ['Player', 'Performance_Gls']):
    goals_2023_24 = df1[['Player', 'Performance_Gls']].copy().rename(columns={'Performance_Gls': 'Goals_2023_24'})
    goals_2024_25 = df2[['Player', 'Performance_Gls']].copy().rename(columns={'Performance_Gls': 'Goals_2024_25'})
    goals_comparison = pd.merge(goals_2023_24, goals_2024_25, on='Player', how='inner')
    goals_comparison_cleaned = goals_comparison.dropna().iloc[:-2].copy()
    goals_comparison_cleaned_filtered = goals_comparison_cleaned[(goals_comparison_cleaned['Goals_2023_24'] > 0) & (goals_comparison_cleaned['Goals_2024_25'] > 0)].copy()
    st.dataframe(goals_comparison_cleaned_filtered)

    goals_comparison_melted = goals_comparison_cleaned_filtered.melt('Player', var_name='Season', value_name='Goals')
    fig11, ax11 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Goals', y='Player', hue='Season', data=goals_comparison_melted.sort_values('Goals', ascending=False), ax=ax11)
    ax11.set_title('Comparison of Goals Scored Per Player (2023/24 vs 2024/25) - Players with Goals in Both Seasons')
    ax11.set_xlabel('Goals Scored')
    ax11.set_ylabel('Player')
    ax11.legend(title='Season')
    plt.tight_layout()
    for p in ax11.patches:
        width = p.get_width()
        ax11.text(width, p.get_y() + p.get_height()/2.,
                 '{:1.0f}'.format(width),
                 ha='left', va='center')
    st.pyplot(fig11)
    plt.close(fig11)
else:
    st.write("Required columns for Goals Comparison not found in one or both seasons' data.")


st.subheader('Assists Comparison')
if all(col in df1.columns for col in ['Player', 'Performance_Ast']) and all(col in df2.columns for col in ['Player', 'Performance_Ast']):
    assists_2023_24 = df1[['Player', 'Performance_Ast']].copy().rename(columns={'Performance_Ast': 'Assists_2023_24'})
    assists_2024_25 = df2[['Player', 'Performance_Ast']].copy().rename(columns={'Performance_Ast': 'Assists_2024_25'})
    assists_comparison = pd.merge(assists_2023_24, assists_2024_25, on='Player', how='inner')
    assists_comparison_cleaned = assists_comparison.dropna().iloc[:-2].copy()
    st.dataframe(assists_comparison_cleaned)

    assists_comparison_melted = assists_comparison_cleaned.melt('Player', var_name='Season', value_name='Assists')
    fig12, ax12 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Assists', y='Player', hue='Season', data=assists_comparison_melted.sort_values('Assists', ascending=False), ax=ax12)
    ax12.set_title('Comparison of Assists Per Player (2023/24 vs 2024/25)')
    ax12.set_xlabel('Assists')
    ax12.set_ylabel('Player')
    ax12.legend(title='Season')
    plt.tight_layout()
    for p in ax12.patches:
        width = p.get_width()
        ax12.text(width, p.get_y() + p.get_height()/2.,
                 '{:1.0f}'.format(width),
                 ha='left', va='center')
    st.pyplot(fig12)
    plt.close(fig12)
else:
    st.write("Required columns for Assists Comparison not found in one or both seasons' data.")


st.subheader('Goals + Assists Comparison')
if all(col in df1.columns for col in ['Player', 'G+A']) and all(col in df2.columns for col in ['Player', 'G+A']):
    ga_2023_24 = df1[['Player', 'G+A']].copy().rename(columns={'G+A': 'G+A_2023_24'})
    ga_2024_25 = df2[['Player', 'G+A']].copy().rename(columns={'G+A': 'G+A_2024_25'})
    ga_comparison = pd.merge(ga_2023_24, ga_2024_25, on='Player', how='inner')
    ga_comparison_cleaned = ga_comparison.dropna().iloc[:-2].copy()
    st.dataframe(ga_comparison_cleaned)

    ga_comparison_melted = ga_comparison_cleaned.melt('Player', var_name='Season', value_name='G+A')
    fig13, ax13 = plt.subplots(figsize=(12, 8))
    sns.barplot(x='G+A', y='Player', hue='Season', data=ga_comparison_melted.sort_values('G+A', ascending=False), ax=ax13)
    ax13.set_title('Comparison of Goals + Assists Per Player (2023/24 vs 2024/25)')
    ax13.set_xlabel('Goals + Assists')
    ax13.set_ylabel('Player')
    ax13.legend(title='Season')
    plt.tight_layout()
    for p in ax13.patches:
        width = p.get_width()
        ax13.text(width, p.get_y() + p.get_height()/2.,
                 '{:1.0f}'.format(width),
                 ha='left', va='center')
    st.pyplot(fig13)
    plt.close(fig13)
else:
    st.write("Required columns for Goals + Assists Comparison not found in one or both seasons' data.")
