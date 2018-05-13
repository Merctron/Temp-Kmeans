import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as sn
from scipy.stats import skew, kurtosis
filename = 'tagged_data_01_2017.csv' # csv file to load
data = pd.read_csv(filename, delimiter=';', keep_default_na=False, na_values=['NaN'], dtype='unicode') # load data to pandas DataFrame

data.set_index('time', inplace = True) # set column 'time' as index

data.index = pd.to_datetime(data.index, format='%d/%m/%y %H:%M') # convert index to datetime objects
data = data.apply(pd.to_numeric) # convert data id DataFrame to numbers
columns = list(data.columns.values) # save DataFrame header to list
# print(data[columns[0]]) # print timeserie stored in DataFrame data indexed by list of columns names (header)
#data[columns[0][:]].plot() # plot data with indexing by list of columns names (header)

data = data[data.index.dayofweek < 5] #returns working days (can be used for specific days 0-6)
data = data.filter(regex=("^.*temperature.*$"))

daymeans = data.groupby(data.index.day).agg(lambda x: np.nanmean(x[x<100]))
daystds = data.groupby(data.index.day).agg(lambda x: np.nanstd(x[x<100]))
# dayamps= data.groupby(data.index.day).agg(lambda x: np.nanmedian(x[x<100]) - np.nanmean(x[x<100]))
dayamps= data.groupby(data.index.month).agg(lambda x: skew(x[x<100]))

indoordaymeans = daymeans.filter(regex=("indoor_air_temperature.*"))
indoordaystds = daystds.filter(regex=("indoor_air_temperature.*"))
indoordayamps = dayamps.filter(regex=("indoor_air_temperature.*"))
np_in_temp_means_nan = np.reshape(indoordaymeans.as_matrix(), -1)
np_in_temp_stds_nan = np.reshape(indoordaystds.as_matrix(), -1)
np_in_temp_amps_nan = np.reshape(indoordayamps.as_matrix(), -1)
np_in_temp_means = np_in_temp_means_nan[~np.isnan(np_in_temp_means_nan)] #filtering Nan
np_in_temp_stds = np_in_temp_stds_nan[~np.isnan(np_in_temp_stds_nan)]
np_in_temp_amps = np_in_temp_amps_nan[~np.isnan(np_in_temp_amps_nan)]

outdoordaymeans = daymeans.filter(regex=("outdoor_air_temperature.*"))
outdoordaystds = daystds.filter(regex=("outdoor_air_temperature.*"))
outdoordayamps = dayamps.filter(regex=("outdoor_air_temperature.*"))
np_out_temp_means_nan = np.reshape(outdoordaymeans.as_matrix(), -1)
np_out_temp_stds_nan = np.reshape(outdoordaystds.as_matrix(), -1)
np_out_temp_amps_nan = np.reshape(outdoordayamps.as_matrix(), -1)
np_out_temp_means = np_out_temp_means_nan[~np.isnan(np_out_temp_means_nan)]
np_out_temp_stds = np_out_temp_stds_nan[~np.isnan(np_out_temp_stds_nan)]
np_out_temp_amps = np_out_temp_amps_nan[~np.isnan(np_out_temp_amps_nan)]

supplydaymeans = daymeans.filter(regex=("supply_water_temperature.*"))
supplydaystds = daystds.filter(regex=("supply_water_temperature.*"))
supplydayamps = dayamps.filter(regex=("supply_water_temperature.*"))
np_supply_temp_means_nan = np.reshape(supplydaymeans.as_matrix(), -1)
np_supply_temp_stds_nan = np.reshape(supplydaystds.as_matrix(), -1)
np_supply_temp_amps_nan = np.reshape(supplydayamps.as_matrix(), -1)
np_supply_temp_means = np_supply_temp_means_nan[~np.isnan(np_supply_temp_means_nan)]
np_supply_temp_stds = np_supply_temp_stds_nan[~np.isnan(np_supply_temp_stds_nan)]
np_supply_temp_amps = np_supply_temp_amps_nan[~np.isnan(np_supply_temp_amps_nan)]

returndaymeans = daymeans.filter(regex=("return_water_temperature.*"))
returndaystds = daystds.filter(regex=("return_water_temperature.*"))
returndayamps = dayamps.filter(regex=("return_water_temperature.*"))
np_return_temp_means_nan = np.reshape(returndaymeans.as_matrix(), -1)
np_return_temp_stds_nan = np.reshape(returndaystds.as_matrix(), -1)
np_return_temp_amps_nan = np.reshape(returndayamps.as_matrix(), -1)
np_return_temp_means = np_return_temp_means_nan[~np.isnan(np_return_temp_means_nan)]
np_return_temp_stds = np_return_temp_stds_nan[~np.isnan(np_return_temp_stds_nan)]
np_return_temp_amps = np_return_temp_amps_nan[~np.isnan(np_return_temp_amps_nan)]

warmdaymeans = daymeans.filter(regex=("warm_service_water_temperature.*"))
warmdaystds = daystds.filter(regex=("warm_service_water_temperature.*"))
warmdayamps = dayamps.filter(regex=("warm_service_water_temperature.*"))
np_warm_temp_means_nan = np.reshape(warmdaymeans.as_matrix(), -1)
np_warm_temp_stds_nan = np.reshape(warmdaystds.as_matrix(), -1)
np_warm_temp_amps_nan = np.reshape(warmdayamps.as_matrix(), -1)
np_warm_temp_means = np_warm_temp_means_nan[~np.isnan(np_warm_temp_means_nan)]
np_warm_temp_stds = np_warm_temp_stds_nan[~np.isnan(np_warm_temp_stds_nan)]
np_warm_temp_amps = np_warm_temp_amps_nan[~np.isnan(np_warm_temp_amps_nan)]

bins_mean = np.linspace(-30, 100, 260)
bins_std = np.linspace(0, 20, 400)

hist_in_means, bins_in_means = np.histogram(np_in_temp_means, bins=bins_mean)
plt.hist(np_in_temp_means, bins=bins_mean)
plt.hist(np_out_temp_means, bins=bins_mean)
plt.hist(np_supply_temp_means, bins=bins_mean)
plt.hist(np_return_temp_means, bins=bins_mean)
plt.figure()
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_in_temp_means), np.nanstd(np_in_temp_means)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_out_temp_means), np.nanstd(np_out_temp_means)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_supply_temp_means), np.nanstd(np_supply_temp_means)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_return_temp_means), np.nanstd(np_return_temp_means)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_warm_temp_means), np.nanstd(np_warm_temp_means)))
plt.gca().set_prop_cycle(None)
plt.hist(np_in_temp_means, bins=bins_mean, density=True)
plt.hist(np_out_temp_means, bins=bins_mean, density=True)
plt.hist(np_supply_temp_means, bins=bins_mean, density=True)
plt.hist(np_return_temp_means, bins=bins_mean, density=True)
plt.hist(np_warm_temp_means, bins=bins_mean, density=True)

plt.figure()
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_in_temp_amps), np.nanstd(np_in_temp_amps)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_out_temp_amps), np.nanstd(np_out_temp_amps)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_supply_temp_amps), np.nanstd(np_supply_temp_amps)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_return_temp_amps), np.nanstd(np_return_temp_amps)))
plt.plot(range(-15,100), sn.pdf(range(-15,100), np.nanmean(np_warm_temp_amps), np.nanstd(np_warm_temp_amps)))
plt.gca().set_prop_cycle(None)
plt.hist(np_in_temp_amps, bins=bins_mean, density=True)
plt.hist(np_out_temp_amps, bins=bins_mean, density=True)
plt.hist(np_supply_temp_amps, bins=bins_mean, density=True)
plt.hist(np_return_temp_amps, bins=bins_mean, density=True)
plt.hist(np_warm_temp_amps, bins=bins_mean, density=True)



plt.figure()

# hist_in_stds, bins_in_stds = np.histogram(np_in_temp_stds, bins=bins_std)
plt.hist(np_in_temp_stds, bins=bins_std)
plt.hist(np_out_temp_stds, bins=bins_std)
plt.hist(np_supply_temp_stds, bins=bins_std)
plt.hist(np_return_temp_stds, bins=bins_std)
plt.hist(np_warm_temp_stds, bins=bins_std)


temp_daymeans = daymeans.filter(regex=("^.*temperature.*$"))
temp_daystds = daystds.filter(regex=("^.*temperature.*$"))
temp_dayamps = dayamps.filter(regex=("^.*temperature.*$"))
incorrect = [0,0]
outcorrect = [0,0]
supcorrect = [0,0]
retcorrect = [0,0]
sercorrect = [0,0]

infalse = 0
outfalse = 0
supfalse = 0
retfalse = 0
serfalse = 0

allfalse = 0
allvar = 0
for column in temp_daymeans:
	try:
		temp_daymeans_col = temp_daymeans[column].as_matrix().reshape(-1)
		temp_daystds_col = temp_daystds[column].as_matrix().reshape(-1)
		temp_dayamps_col = temp_dayamps[column].as_matrix().reshape(-1)
	except:
		continue
	allvar += 1
	indoor_norm = sn.pdf(np.mean(np_in_temp_means), np.mean(np_in_temp_means), np.std(np_in_temp_means))
	outdoor_norm = sn.pdf(np.mean(np_out_temp_means), np.mean(np_out_temp_means), np.std(np_out_temp_means))
	return_norm = sn.pdf(np.mean(np_return_temp_means), np.mean(np_return_temp_means), np.std(np_return_temp_means))
	supply_norm = sn.pdf(np.mean(np_supply_temp_means), np.mean(np_supply_temp_means), np.std(np_supply_temp_means))
	warm_norm = sn.pdf(np.mean(np_warm_temp_means), np.mean(np_warm_temp_means), np.std(np_warm_temp_means))

	indoor_norm_std = sn.pdf(np.mean(np_in_temp_stds), np.mean(np_in_temp_stds), np.std(np_in_temp_stds))
	outdoor_norm_std = sn.pdf(np.mean(np_out_temp_stds), np.mean(np_out_temp_stds), np.std(np_out_temp_stds))
	return_norm_std = sn.pdf(np.mean(np_return_temp_stds), np.mean(np_return_temp_stds), np.std(np_return_temp_stds))
	supply_norm_std = sn.pdf(np.mean(np_supply_temp_stds), np.mean(np_supply_temp_stds), np.std(np_supply_temp_stds))
	warm_norm_std = sn.pdf(np.mean(np_warm_temp_stds), np.mean(np_warm_temp_stds), np.std(np_warm_temp_stds))

	indoor_norm_amp = sn.pdf(np.mean(np_in_temp_amps), np.mean(np_in_temp_amps), np.std(np_in_temp_amps))
	outdoor_norm_amp = sn.pdf(np.mean(np_out_temp_amps), np.mean(np_out_temp_amps), np.std(np_out_temp_amps))
	return_norm_amp = sn.pdf(np.mean(np_return_temp_amps), np.mean(np_return_temp_amps), np.std(np_return_temp_amps))
	supply_norm_amp = sn.pdf(np.mean(np_supply_temp_amps), np.mean(np_supply_temp_amps), np.std(np_supply_temp_amps))
	warm_norm_amp = sn.pdf(np.mean(np_warm_temp_amps), np.mean(np_warm_temp_amps), np.std(np_warm_temp_amps))

	temp_daymeans_col_mean = np.nanmean(temp_daymeans_col)
	temp_daystds_col_mean = np.nanmean(temp_daystds_col)
	temp_dayamps_col_mean = np.nanmean(temp_dayamps_col)

	prob_indoor_temp_m = sn.pdf(temp_daymeans_col_mean, np.mean(np_in_temp_means), np.std(np_in_temp_means))/indoor_norm
	prob_outdoor_temp_m = sn.pdf(temp_daymeans_col_mean, np.mean(np_out_temp_means), np.std(np_out_temp_means))/outdoor_norm
	prob_return_temp_m = sn.pdf(temp_daymeans_col_mean, np.mean(np_return_temp_means), np.std(np_return_temp_means))/return_norm
	prob_supply_temp_m = sn.pdf(temp_daymeans_col_mean, np.mean(np_supply_temp_means), np.std(np_supply_temp_means))/supply_norm
	prob_warm_temp_m = sn.pdf(temp_daymeans_col_mean, np.mean(np_warm_temp_means), np.std(np_warm_temp_means))/warm_norm

	prob_indoor_temp_s = sn.pdf(temp_daystds_col_mean, np.mean(np_in_temp_stds), np.std(np_in_temp_stds))/indoor_norm_std
	prob_outdoor_temp_s = sn.pdf(temp_daystds_col_mean, np.mean(np_out_temp_stds), np.std(np_out_temp_stds))/outdoor_norm_std
	prob_return_temp_s = sn.pdf(temp_daystds_col_mean, np.mean(np_return_temp_stds), np.std(np_return_temp_stds))/return_norm_std
	prob_supply_temp_s = sn.pdf(temp_daystds_col_mean, np.mean(np_supply_temp_stds), np.std(np_supply_temp_stds))/supply_norm_std
	prob_warm_temp_s = sn.pdf(temp_daystds_col_mean, np.mean(np_warm_temp_stds), np.std(np_warm_temp_stds))/warm_norm_std

	prob_indoor_temp_a = sn.pdf(temp_dayamps_col_mean, np.mean(np_in_temp_amps), np.std(np_in_temp_amps))/indoor_norm_amp
	prob_outdoor_temp_a = sn.pdf(temp_dayamps_col_mean, np.mean(np_out_temp_amps), np.std(np_out_temp_amps))/outdoor_norm_amp
	prob_return_temp_a = sn.pdf(temp_dayamps_col_mean, np.mean(np_return_temp_amps), np.std(np_return_temp_amps))/return_norm_amp
	prob_supply_temp_a = sn.pdf(temp_dayamps_col_mean, np.mean(np_supply_temp_amps), np.std(np_supply_temp_amps))/supply_norm_amp
	prob_warm_temp_a = sn.pdf(temp_dayamps_col_mean, np.mean(np_warm_temp_amps), np.std(np_warm_temp_amps))/warm_norm_amp


	probabilities_s = np.array([prob_indoor_temp_s, 
					 prob_outdoor_temp_s, 
					 prob_return_temp_s, 
					 prob_supply_temp_s, 
					 prob_warm_temp_s])

	probabilities_m = np.array([prob_indoor_temp_m, 
					 prob_outdoor_temp_m, 
					 prob_return_temp_m, 
					 prob_supply_temp_m, 
					 prob_warm_temp_m])

	probabilities_a = np.array([prob_indoor_temp_a, 
					 prob_outdoor_temp_a, 
					 prob_return_temp_a, 
					 prob_supply_temp_a, 
					 prob_warm_temp_a])

	# probabilities = [prob_indoor_temp_m*prob_indoor_temp_s*prob_indoor_temp_a, 
	# 				 prob_outdoor_temp_m*prob_outdoor_temp_s*prob_outdoor_temp_a, 
	# 				 prob_return_temp_m*prob_return_temp_s*prob_return_temp_a, 
	# 				 prob_supply_temp_m*prob_supply_temp_s*prob_supply_temp_a, 
	# 				 prob_warm_temp_m*prob_warm_temp_s*prob_warm_temp_a]
	names= ['indoor', 'outdoor', 'return', 'supply', 'warm_service']
	sort = np.argsort(probabilities_m*probabilities_a*probabilities_s)

	if 'indoor' in column:
		incorrect[0 if names[sort[-1]] in column else 1] += 1
	if 'outdoor' in column:
		outcorrect[0 if names[sort[-1]] in column else 1] += 1
	if 'supply' in column:
		supcorrect[0 if names[sort[-1]] in column else 1] += 1
	if 'return' in column:
		retcorrect[0 if names[sort[-1]] in column else 1] += 1
	if 'service' in column:
		sercorrect[0 if names[sort[-1]] in column else 1] += 1

	if not names[sort[-1]] in column:
		if names[sort[-1]] == 'indoor':
			infalse += 1
		if names[sort[-1]] == 'oudoor':
				outfalse += 1
		if names[sort[-1]] == 'supply':
				supfalse += 1
		if names[sort[-1]] == 'return':
				retfalse += 1
		if names[sort[-1]] == 'service':
				serfalse += 1
	allfalse += 1	
	# print(column[:20], '\t', names[sort[-1]], '%f' % probabilities[sort[-1]], 'correct' if names[sort[-1]] in column else 'false')

    
# output bayesian theorem
print('\n')
print('indoor ','\t', '%.1f' % (incorrect[0]/sum(incorrect)*100), '\t', '%.1f' % (infalse/(allfalse-sum(incorrect))*100),'\t', '%0.2f' % ((incorrect[0]/sum(incorrect)*sum(incorrect)/allvar)/((incorrect[0]/sum(incorrect)*sum(incorrect)/allvar)+infalse/(allfalse-sum(incorrect))*(1-sum(incorrect)/allvar))))
print('outdoor', '\t', '%.1f' % (outcorrect[0]/sum(outcorrect)*100), '\t', '%.1f' % (outfalse/(allfalse-sum(outcorrect))*100),'\t', '%0.2f' % ((outcorrect[0]/sum(outcorrect)*sum(outcorrect)/allvar)/((outcorrect[0]/sum(outcorrect)*sum(outcorrect)/allvar)+outfalse/(allfalse-sum(outcorrect))*(1-sum(outcorrect)/allvar))))
print('supply ', '\t', '%.1f' % (supcorrect[0]/sum(supcorrect)*100), '\t', '%.1f' % (supfalse/(allfalse-sum(supcorrect))*100),'\t', '%0.2f' % ((supcorrect[0]/sum(supcorrect)*sum(supcorrect)/allvar)/((supcorrect[0]/sum(supcorrect)*sum(supcorrect)/allvar)+supfalse/(allfalse-sum(supcorrect))*(1-sum(supcorrect)/allvar))))
print('return ', '\t', '%.1f' % (retcorrect[0]/sum(retcorrect)*100), '\t', '%.1f' % (retfalse/(allfalse-sum(retcorrect))*100),'\t', '%0.2f' % ((retcorrect[0]/sum(retcorrect)*sum(retcorrect)/allvar)/((retcorrect[0]/sum(retcorrect)*sum(retcorrect)/allvar)+retfalse/(allfalse-sum(retcorrect))*(1-sum(retcorrect)/allvar))))
print('service', '\t', '%.1f' % (sercorrect[0]/sum(sercorrect)*100), '\t', '%.1f' % (serfalse/(allfalse-sum(sercorrect))*100),'\t', '%0.2f' % ((sercorrect[0]/sum(sercorrect)*sum(sercorrect)/allvar)/((sercorrect[0]/sum(sercorrect)*sum(sercorrect)/allvar)+serfalse/(allfalse-sum(sercorrect))*(1-sum(sercorrect)/allvar)+10e-8)))

# example how to call function over data in dataframe (evaluation of mean value of columns)


# # one way - using pandas function
# def function_over_data(column): 
# 	return column.mean()
	
# dataret = data.apply(function_over_data)
# print(dataret)

# # second way - using for loop, slower, beter when you want different output then pandas series
# dataret = pd.Series()
# for idx, val in data.iteritems():
# 	dataret[idx] = val.mean()

# print(dataret)

# data[columns[0][:]].plot() # plot data with indexing by list of columns names (header)
plt.show()
