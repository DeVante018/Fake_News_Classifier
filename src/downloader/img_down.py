import urllib.request, urllib.error
import argparse
import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np
import sys


url = "https://preview.redd.it/l0ga0tug17k31.jpg?width=320&crop=smart&auto=webp&s=ef3396b88eba2a0ed976b0d6b6ee9ea29733d32d"
try:
    conn = urllib.request.urlopen(url)
except urllib.error.HTTPError as e:
    # Return code error (e.g. 404, 501, ...)
    # ...
    print('HTTPError: {}'.format(e.code))
except urllib.error.URLError as e:
    # Not an HTTP-specific error (e.g. connection refused)
    # ...
    print('URLError: {}'.format(e.reason))
else:
    print("good")


df = pd.read_csv(r'C:\Users\ccastano\Desktop\multimodal_test_public_2.tsv', sep="\t")
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

pbar = tqdm(total=len(df))


for index, row in df.iterrows():
    if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
        image_url = row["image_url"]
        try:
            conn = urllib.request.urlopen(image_url)
        except urllib.error.HTTPError as e:
            # Return code error (e.g. 404, 501, ...)
            # ...
            print('HTTPError: {}'.format(e.code))
        except urllib.error.URLError as e:
            # Not an HTTP-specific error (e.g. connection refused)
            # ...
            print('URLError: {}'.format(e.reason))
        else:
            print("good")

            print(image_url)

            urllib.request.urlretrieve(image_url, r"C:/Users/ccastano/Desktop/basedata/testing/misleading/" + row["id"] + ".jpg")

            pbar.update(1)
print("done")