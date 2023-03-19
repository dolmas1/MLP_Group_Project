import re
import pandas as pd

df = pd.read_csv(f'../data/twitter_hatespeech/twitter_hatespeech.tsv', sep='\t')

def clean_text(txt):
    txt_clean = re.sub(r"\[NEWLINE\]", ' ', txt)                                  # remove NEWLINE
    txt_clean = re.sub(r"’", "'", txt_clean)                                      # silly quote marks
    txt_clean = re.sub(r"https?:\/\/(\d|[a-z]|[A-Z]|\.|\/)+", '', txt_clean)      # strip URLs
    txt_clean = re.sub(r"\&amp", '&', txt_clean)                                  # &amp
    txt_clean = re.sub(r"\_", ' ', txt_clean)                                     # replace underscores with space
    txt_clean = re.sub(r"[^(\w|\s|(\'\,\-\_\.\%\!\$\(\)\:))]", '', txt_clean)     # remove all other non-standard chars
    txt_clean = re.sub(r"^\s+", '', txt_clean)                                    # remove leading spaces
    txt_clean = re.sub(r"\.\.+", '.. ', txt_clean)                                # multiple full stops
    txt_clean = re.sub(r" +", ' ', txt_clean)                                     # replace multiple spaces with single
    
    return txt_clean.lower()

df.text = [clean_text(txt) for txt in df.text]
df.to_csv(f'../data/twitter_hatespeech/twitter_hatespeech_clean.tsv', sep='\t', index=False)