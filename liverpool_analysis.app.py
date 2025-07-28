import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text 

# Load and clean df1
df1 = pd.read_html('https://fbref.com/en/squads/822bd0ba/2023-2024/Liverpool-Stats', attrs={"id":"stats_standard_9"})[0]
df1.columns = ['_'.join(col).strip() for col in df1.columns.values]
df1.columns = [col.replace('Unnamed: 0_level_0_', '') if 'Unnamed: 0_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 1_level_0_', '') if 'Unnamed: 1_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 2_level_0_', '') if 'Unnamed: 2_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 3_level_0_', '') if 'Unnamed: 3_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 4_level_0_', '') if 'Unnamed: 4_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 5_level_0_', '') if 'Unnamed: 5_level_0_' in col else col for col in df1.columns]
df1.columns = [col.replace('Unnamed: 6_level_0_', '') if 'Unnamed: 6_level_0_' in col else col for col in df1.columns]
pos_col_index = df1.columns.get_loc('Pos')
for col in df1.columns[pos_col_index + 1:]:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
last_column_name = df1.columns[-1]
df1 = df1.drop(columns=[last_column_name])
df1['Nation'] = df1['Nation'].str[2:]
df1['Performance_npxG+A'] = df1['Performance_G+A'] - df1['Performance_PK']


# Load and clean df2
df2 = pd.read_html('https://fbref.com/en/squads/822bd0ba/2024-2025/Liverpool-Stats', attrs={"id":"stats_standard_9"})[0]
df2.columns = ['_'.join(col).strip() if col[0] else col[1].strip() for col in df2.columns.values]
df2.columns = [col.replace('Unnamed: 0_level_0_', '') if 'Unnamed: 0_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 1_level_0_', '') if 'Unnamed: 1_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 2_level_0_', '') if 'Unnamed: 2_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 3_level_0_', '') if 'Unnamed: 3_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 4_level_0_', '') if 'Unnamed: 4_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 5_level_0_', '') if 'Unnamed: 5_level_0_' in col else col for col in df2.columns]
df2.columns = [col.replace('Unnamed: 6_level_0_', '') if 'Unnamed: 6_level_0_' in col else col for col in df2.columns]
pos_col_index_df2 = df2.columns.get_loc('Pos')
for col in df2.columns[pos_col_index_df2 + 1:]:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
last_column_name_df2 = df2.columns[-1]
df2 = df2.drop(columns=[last_column_name_df2])
df2['Nation'] = df2['Nation'].str[2:]
df2['Performance_npxG+A'] = df2['Performance_G+A'] - df2['Performance_PK']

st.write("Data loaded and cleaned successfully.")
