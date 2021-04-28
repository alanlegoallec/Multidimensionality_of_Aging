import pandas as pd

def read_physical_activity_data(**kwargs):
    df_age_new = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').set_index('id').drop(columns = ['Abdominal_images_quality', 'instance'])
    df_age_new = df_age_new.rename(columns = {'Age' : 'Age when attended assessment centre'})
    df_physical = pd.read_csv('/n/groups/patel/Alan/Aging/TimeSeries/series/PhysicalActivity/90001/features/PA_all_features.csv').set_index('id').drop(columns = ['eid'])
    df = df_age_new.join(df_physical)
    return df.dropna()
