import os 
import glob
import pandas as pd
import json
from sklearn.metrics import r2_score, f1_score

final_dict = {}
for target in ['Sex','Age']:
	final_dict[target] = {}
	for dataset in ['abdominal_composition', 'brain_grey_matter_volumes', 'brain_subcortical_volumes']:
		final_dict[target][dataset] = {}
		for model in [ 'random_forest', 'xgboost', 'lightgbm', 'neural_network'] :
			print('target : ', target, 'model : ', model, 'dataset : ', dataset)
			list_files = [f for f in glob.glob("/n/groups/patel/samuel/HMS-Aging/aging/predictions/*.csv") if target in f and model in f and dataset in f]
			
			list_df = [pd.read_csv(f) for f in list_files]
			final_df = pd.concat(list_df)
			final_df.to_csv('/n/groups/patel/samuel/HMS-Aging/aging/final_predictions/' + target + '_'+ model + '_' + dataset) 
			print(final_df.head())	
			if target == 'Sex':
				score = f1_score(final_df['real'], final_df['predictions'])
			elif target == 'Age':
				score = r2_score(final_df['real'], final_df['predictions'])
			final_dict[target][dataset][model] = score
with open('score.json', 'w') as fp:
	json.dump(final_dict, fp)				
