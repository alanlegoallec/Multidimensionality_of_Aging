"""
Created on Feb 06 2020

@author: Sasha Collin
"""
from TS_ressources import *
import time

# initial time
t0 = time.time()

hp = hyperparameters(sys.argv)
version = hp.get_version(remove_non_relevant_param=False)
print(version)

# Loading data
DF = DataFetcher(hp.target, hp.TS_type)
# partition of the data between train and val/test
labeled_data = DF.get_data()  # dictionary

# Model building
model = TS_model(hp, labeled_data)

# Compile
model.compile(optimizer=Adam, loss='mean_squared_error', metrics=[R_squared, root_mean_squared_error])

# Fit data
history = model.fit(verbose=2, callbacks_metric='R_squared')

# final time
t1 = convert_time(time.time() - t0)

# Saving performances
with open('../data/performances/' + hp.target + '_' + hp.TS_type + '_' + hp.model_type + '_' + hp.sub_model_type + '.csv', 'a') as file:
    writer = csv.writer(file)

    columns = []
    values = []
    for h in hp.__dict__:
        columns.append(h)
        values.append(hp.__dict__[h])
    # columns.append('r2_train')
    columns.append('max_r2_train')
    # values.append(history.history['R_squared'][-1])
    values.append(max(history.history['R_squared']))
    # columns.append('r2_val')
    columns.append('max_r2_val')
    # values.append(history.history['val_R_squared'][-1])
    values.append(max(history.history['val_R_squared']))

    # add time
    columns.append('time')
    values.append(t1)

    if file.tell() == 0:
        writer.writerow(columns)

    writer.writerow(values)
