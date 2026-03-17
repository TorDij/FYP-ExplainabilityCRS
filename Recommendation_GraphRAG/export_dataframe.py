import pandas as pd
import dataframe_image as dfi

# Load DataFrame
community_df = pd.read_parquet(r"/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/test_dialogue/artifacts/create_final_community_reports.parquet")

# Select relevant columns
community_df = community_df[['human_readable_id', 'title', 'summary', 'full_content', 'rank_explanation', 'findings', 'full_content_json']]

# Choose record
row_index = 0  
record = community_df.loc[row_index].to_frame().T 

# Set display options for better formatting
pd.set_option('display.max_colwidth', None)  # Prevents text truncation
pd.set_option('display.width', 1000)

# Style DataFrame
df_styled = record.style.set_properties(**{
    'text-align': 'left',
    'white-space': 'pre-wrap'
}).set_table_styles([
    {'selector': 'th, td', 'props': [('max-width', '900px'), ('word-wrap', 'break-word')]}  # Expands columns
])

# Save as an image using Chrome rendering for higher quality
dfi.export(df_styled, 'record.png', table_conversion='chrome', dpi=600)

print("Image saved as record.png")
