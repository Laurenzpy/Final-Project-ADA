import json
import pandas as pd
import re

#load full_emoji.csv into a pandas dataframe
emoji_df = pd.read_csv('full_emoji.csv')

#only keep first 4 columns of the dataframe
emoji_df = emoji_df.iloc[:, :4]


#add english meanings from the emojis.json file to the dataframe

# Load emojis.json (list of dicts)
with open("emojis.json", "r", encoding="utf-8") as f:
    emojis_data = json.load(f)

emoji_meanings = pd.json_normalize(emojis_data)

# Keep the columns you actually want to bring onto emoji_df
emoji_meanings = emoji_meanings[["unicode", "keywords", "definition", "shortcode"]].copy()

# Make keywords nicer to read (optional)
emoji_meanings["keywords"] = emoji_meanings["keywords"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else x
)

print(emoji_meanings.head())
print(emoji_meanings.dtypes)

# Merge the two dataframes on the 'unicode' column
merged_emoji_df = pd.merge(emoji_df, emoji_meanings, on="unicode", how="left")

# in definition column, remove any text after the first period (including the period)
merged_emoji_df["definition"] = merged_emoji_df["definition"].apply(
    lambda x: x.split('.', 1)[0] if isinstance(x, str) else x
)


#print(merged_emoji_df.shape)
#print(merged_emoji_df.columns)
#print(merged_emoji_df.dtypes)
#print(merged_emoji_df.head())

#create csv file from merged_emoji_df of first 5 rows only
merged_emoji_df.head(5).to_csv('merged_emoji_sample.csv', index=False)

