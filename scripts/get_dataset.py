import pandas as pd
import requests
import json
import time
import sys
from datetime import datetime, timedelta

from tqdm import tqdm

# This script is direct conversion from jupyter notebook.

N_DAYS = 120
TIME_WINDOW = 14

def clean_data_fn(data):
    data = data['allFranceDataByDate']
    try:
        return list(map(lambda x: (x['nom'], x['hospitalises']), data))
    except:
        return None

def main():
    n = 120
    _date = datetime.now()

    dates = []
    for e in range(n):
        _date = _date - timedelta(days=1)
        dates.append(_date)

    dates = [x.strftime("%Y-%m-%d") for x in dates]

    data = []
    errors = []
    for d in tqdm(dates):
        try:
            url = "https://coronavirusapi-france.now.sh/AllDataByDate?date={}".format(d)
            r = requests.get(url).json()
            data.append(r)
            time.sleep(1)
        except json.JSONDecodeError:
            errors.append(d)
        
    clean_data = list(map(clean_data_fn, data))
    names = [x[0] for x in clean_data[0] if x != None]
    values = [[x[1] for x in sublist] for sublist in clean_data if sublist != None]
    
    df_list = []
    for i, d in enumerate(data):
        df = pd.DataFrame(d['allFranceDataByDate'])[['nom', 'hospitalises']]
        df.set_index('nom', inplace=True)
        df.columns = ['t-{}'.format(i)]
        df_list.append(df)
    
    for i, _df in enumerate(df_list[1:]):
        try:
            df = pd.concat([df, _df], axis=1)
        except:
            # Ignore malformed data for sake of simplicity
            pass
        
    # Get all values for dataset
    n = df.shape[1]
    time_window = 14
    max_df_size = df.shape[1] - 14 # For sanity check

    all_labels = []
    all_values = []
    for idx in df.index:
        serie = df.loc[idx]
        serie_list = []
        for day in range(n):
            if serie.shift(-day).dropna().shape[0] == time_window:
                break
            serie_list.append(serie.shift(-day))
        dpt_df = pd.concat(serie_list, axis=1).T

        dpt_labels = dpt_df.iloc[:, 0].to_numpy()
        dpt_values = dpt_df.iloc[:, 1:time_window + 1].to_numpy()

        discrete_labels = []
        for i, e in enumerate(dpt_labels):
            if e > dpt_values[i][-1]:
                discrete_labels.append(1)
            else:
                discrete_labels.append(-1)

        if len(discrete_labels) <= max_df_size:
            all_labels.append(discrete_labels)
            all_values.append(pd.DataFrame(dpt_values))

    all_labels = [x for sublist in all_labels for x in sublist]
    all_values_df = pd.concat(all_values)
    
    
    
    # Sample dataset
    label_serie = pd.Series(all_labels)
    counts = label_serie.value_counts()

    diff = counts.max() - counts.min()

    n_chunks = 2
    chunk_1 = (diff / n_chunks) - (diff % n_chunks) / n_chunks

    # Get new samples by random and replace
    idx_list = label_serie.loc[label_serie == 1].index
    pos_samples = all_values_df.iloc[idx_list].sample(int(chunk_1), replace=True)
    pos_samples['label'] = 1

    # Remove some neg samples
    idx_list = label_serie.loc[label_serie == -1].index[:int(chunk_1)]
    neg_samples = all_values_df.iloc[idx_list]
    neg_samples['label'] = -1

    sampled_df = pd.concat([pos_samples, neg_samples])
    
    
    # Save dataset
    if len(sys.argv) == 1:
        path = "./"
    else:
        path = sys.argv[1]
    sampled_df.to_csv(path + 'sampled_covid_dataset.csv', index=False)
        
if __name__ == '__main__':
    main()