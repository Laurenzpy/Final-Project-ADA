# %%
import json
import pandas as pd
import re
# %%


# %%
#load full_emoji.csv into a pandas dataframe
emoji_df = pd.read_csv('full_emoji.csv')

#only keep first 4 columns of the dataframe
emoji_df = emoji_df.iloc[:, :4]
# %%


# %%
#add english meanings from the emojis.json file to the dataframe

# Load emojis.json (list of dicts)
with open("emojis.json", "r", encoding="utf-8") as f:
    emojis_data = json.load(f)

emoji_meanings = pd.json_normalize(emojis_data)
# %%

# %%
def first_gloss(senses_list):
    """
    senses_list is like:
    [ {"bn:xxxx": ["gloss1", "gloss2", ...]}, {"bn:yyyy": [...]} ... ]
    Returns the first gloss of the first sense, or None.
    """
    if not isinstance(senses_list, list) or len(senses_list) == 0:
        return None

    first_item = senses_list[0]
    if not isinstance(first_item, dict) or len(first_item) == 0:
        return None

    glosses = next(iter(first_item.values()))  # the list of gloss strings
    if isinstance(glosses, list) and len(glosses) > 0:
        return glosses[0]

    return None

# Extract only the first gloss from each POS column
emoji_meanings["sense_adj_first"]  = emoji_meanings["senses.adjectives"].apply(first_gloss)
emoji_meanings["sense_verb_first"] = emoji_meanings["senses.verbs"].apply(first_gloss)
emoji_meanings["sense_noun_first"] = emoji_meanings["senses.nouns"].apply(first_gloss)

# (Optional) keep only the relevant columns
emoji_meanings_small = emoji_meanings[
    ["unicode", "keywords", "definition", "shortcode", "sense_adj_first", "sense_verb_first", "sense_noun_first"]
].copy()
# %%




# %%
# Keep the columns you actually want to bring onto emoji_df
#emoji_meanings = emoji_meanings[["unicode", "keywords", "definition", "shortcode"]].copy()

# Make keywords nicer to read (optional)
emoji_meanings_small["keywords"] = emoji_meanings["keywords"].apply(
    lambda x: ", ".join(x) if isinstance(x, list) else x
)

#print(emoji_meanings.head())
#print(emoji_meanings.dtypes)

# Merge the two dataframes on the 'unicode' column
merged_emoji_df = pd.merge(emoji_df, emoji_meanings_small, on="unicode", how="left")

# in definition column, remove any text after the first period (including the period)
merged_emoji_df["definition"] = merged_emoji_df["definition"].apply(
    lambda x: x.split('.', 1)[0] if isinstance(x, str) else x
)
# %%

#print(merged_emoji_df.shape)
#print(merged_emoji_df.columns)
#print(merged_emoji_df.dtypes)
#print(merged_emoji_df.head())

merged_emoji_df["shortcode"] = (
    merged_emoji_df["shortcode"]
    .astype(str)
    .str.replace(":", "", regex=False)
)

# %%
#create csv file from merged_emoji_df of first 5 rows only
merged_emoji_df.to_csv('merged_emoji_sample.csv', index=False)#

# after you create merged_emoji_df
merged_emoji_df.to_parquet("merged_emoji_df.parquet", index=False)
print("Saved to data/merged_emoji_df.parquet")