import numpy as np
import pandas as pd
import seaborn as sns

# df = pd.read_csv("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_concat_2025.csv")

df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')

import matplotlib.pyplot as plt
from functools import reduce
pd.set_option('display.max_colwidth', None)
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name']
# heading_2025 = ["FARMER_ID", "FARMER_GENDER", "FARM_ID", "REPORTING_YEAR", "COUNTRY_NAME", "REGION_NAME", "DISTRICT_NAME"]

ques = df.groupby(by='question')
""" creating a dataframe for pest_level"""
# pest = ques.get_group('What was the pest infestation level in your cocoa plots in the last crop cycle? (pest causing most damage to production): Please refer to guideline for the exact definition of each level.')
# pest_dropped = pest.dropna(subset=['answer'])

# pest_dropped.loc[pest_dropped["answer"].str.contains('Heavy', na=False), "answer"] = 3
# pest_dropped.loc[pest_dropped["answer"].str.contains('Presence but medium', na=False), "answer"] = 2
# pest_dropped.loc[pest_dropped["answer"].str.contains('Presence but low', na=False), "answer"] = 1
# pest_dropped.loc[pest_dropped["answer"].str.contains('Very little to no', na=False), "answer"] = 0
# #Pruning.loc[(Pruning["answer"] == "1") or (Pruning["answer"] == '1 || 1') or (Pruning["answer"] == '0 || 1'), "answer"] = 1
# pest_dropped = pest_dropped.rename(columns={'answer':'pest_level'})
# clean_pest = pest_dropped.drop(columns = ['question', "QUESTION_ID"])
# print(clean_pest['pest_level'].value_counts(dropna = False))

# """ creating a dataframe for disease_level """ 
# disease = ques.get_group('What was the disease contamination level in your cocoa plots in the last crop cycle? (disease causing most damage to production): Please refer to guideline for the exact definition of each level.')
# disease_dropped = disease.dropna(subset=['answer'])
# disease_dropped.loc[disease_dropped["answer"].str.contains('Heavy', na=False), "answer"] = 3
# disease_dropped.loc[disease_dropped["answer"].str.contains('Presence but medium', na=False), "answer"] = 2
# disease_dropped.loc[disease_dropped["answer"].str.contains('Presence but low', na=False), "answer"] = 1
# disease_dropped.loc[disease_dropped["answer"].str.contains('Very little to no', na=False), "answer"] = 0
# #Pruning.loc[(Pruning["answer"] == "1") or (Pruning["answer"] == '1 || 1') or (Pruning["answer"] == '0 || 1'), "answer"] = 1
# disease_dropped = disease_dropped.rename(columns={'answer':'disease_level'})
# clean_disease = disease_dropped.drop(columns = ['question',"QUESTION_ID"])
# print(clean_disease['disease_level'].value_counts())

# def inter_quartiling(dfg, column):
#     country = dfg.iloc[1, 3]
#     data = dfg#.loc[~(dfg[column] < 5)]
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75) 
#     IQR = Q3 - Q1
  
#     # Define bounds
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df_outlied = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]   
#     # plt.figure
#     # sns.displot(df_outlied[column], bins=50,kde=False)
#     # plt.title(f'distribution plot without outliers of {column} of {country}')
#     # plt.show()
#     return df_outlied

## creating a dataframe for irrigation
# irrigation = ques.get_group('ONLY FOR ECUADOR, INDONESIA, BRAZIL - What is the average amount of irrigation water applied in all your cocoa plots (m3/ha)?')
# irrigation_dropped = irrigation.dropna(subset=['answer'])
# irrigation_dropped['answer'] = pd.to_numeric(irrigation_dropped['answer'], errors='coerce')
# irrigation_dropped = irrigation_dropped.rename(columns={'answer':'Irrigation_amount'})
# irrigation_dropped = irrigation_dropped.drop(columns = ['question'])
# print(irrigation_dropped.head())
# print(irrigation_dropped['Irrigation_amount'].describe())
# irrigation_dropped.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_irrigation_2025.csv', index=False)

## removing outliers from the irrigation variable
#irrigation_dropped = irrigation_dropped.groupby(by= 'country_name', group_keys=False).apply(lambda dfg: inter_quartiling(dfg, 'Irrigation_amount'))
# print(Irrigation['Irrigation_amount'].describe())
# sns.displot(Irrigation['Irrigation_amount'], bins=50,kde=False)
# plt.title(f'distribution plot of the irrigation of all age groups')
# plt.show()


## Creating numerical variables
numerical_questions = ['What is the total amount of fungicides applied to all your cocoa plots in the last crop cycle? (report in grams if non-liquid, and non-diluted liters if liquid)', 'What is the total amount of insecticide applied to all your cocoa plots in the last crop cycle?  (report in grams if non-liquid, and non-diluted liters if liquid)','What is the total amount of chemical herbicide applied to all your cocoa plots in the last crop cycle? If liquid, please answer the number of non-diluted liters of herbicide used.']
num_variable_names = ['amount_fungicide','amount_insecticide','amount_herbicide']
merging_dataframe = df.iloc[0:1]
merging_dataframe = merging_dataframe.drop(columns = ['question','answer'])
all_heading = heading

for count in range(3):
    df_question = ques.get_group(numerical_questions[count])
    df_question_dropped = df_question.dropna(subset=['answer'])
    df_question_dropped.loc[:,'answer'] = pd.to_numeric(df_question_dropped['answer'], errors='coerce')
    df_question_dropped = df_question_dropped.rename(columns={'answer': num_variable_names[count]})
    df_question_dropped = df_question_dropped.drop(columns = ['question'])
    # print(df_question_dropped.head())
    # dataframe = df_question_dropped.merge(merging_dataframe, how= 'outer', left_on= heading, right_on= heading)
    # merging_dataframe = dataframe
    # print(merging_dataframe.head())
    all_heading.append(num_variable_names[count])
    dfg = df_question_dropped.groupby(by='country_name')
    print(dfg[num_variable_names[count]].describe())
    #merging_dataframe = merging_dataframe.groupby(by= 'country_name', group_keys=False)[all_heading].apply(lambda dfg: inter_quartiling(dfg, num_variable_names[count]))
    # print(merging_dataframe[num_variable_names[count]].describe())
    # sns.displot(merging_dataframe[num_variable_names[count]], bins=50,kde=False)
    # plt.title(f'distribution plot of the {num_variable_names[count]} of all age groups')
    # plt.show()
dfg = merging_dataframe.groupby(by='country_name')
print(dfg.describe())
# print(merging_dataframe.shape)
# ## Merging the dataframe
# data_frames = [merging_dataframe, clean_disease, clean_pest]
# # dropped_pest = clean_pest.drop_duplicates(subset=['olam_farmer_id'])
# # dropped_disease = clean_disease.drop_duplicates(subset=['olam_farmer_id'])
# df_merged = dropped_disease.merge(dropped_pest, how = 'outer', on= heading)
# df_merged_2 = clean_pest.merge(clean_disease, how = 'outer', on= heading_2025)
# # df_merged = reduce(lambda  left, right: pd.merge(left,right, on= heading, how='outer'), data_frames)
# # pesticide = ['fungicide','insecticide','herbicide']
# # for count in range(3):
# #     merging_dataframe.loc[merging_dataframe[num_variable_names[count]] > 0, pesticide[count]] = 1
# print(df_merged_2.head())
# df_merged_2.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_variables_without_irrigation_2025.csv', index=False)

