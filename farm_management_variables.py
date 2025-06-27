import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_concat_2025.csv")

# df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')

from functools import reduce
pd.set_option('display.max_colwidth', None)
# heading = ['olam_farmer_id','farmer_id','Gender','COUNTRY_NAME','region_name','district_name','place_name']
# all_heading = ['olam_farmer_id','farmer_id','Gender','COUNTRY_NAME','region_name','district_name','place_name','QUESTION_DESC','FINAL_ANSWER']
heading_2025 = ["FARMER_ID", "FARMER_GENDER", "FARM_ID", "REPORTING_YEAR", "COUNTRY_NAME", "REGION_NAME", "DISTRICT_NAME"]
countries = ["COTE D'IVOIRE",'GHANA', "NIGERIA","CAMEROON"]
ques = df.groupby(by='QUESTION_DESC')


# ## creating dataframe from organic fertilizer
# org_fertilizer = ques.get_group('Name of the organic fertilizer applied? (commercial name)')
# org_fertilizer.dropna(subset=['FINAL_ANSWER'], inplace=True)
# org_fertilizer = org_fertilizer.assign(compost = org_fertilizer['FINAL_ANSWER'].str.contains('Compost'))
# org_fertilizer['compost'] = org_fertilizer['compost'].fillna(False).astype(bool)
# org_fertilizer['compost'] = org_fertilizer['compost'].astype(int)
# Frame_compost = org_fertilizer.drop(columns = ['QUESTION_DESC','FINAL_ANSWER'])


# ## creating a dataframe for manure
# Fertilizer = ques.get_group('Name of the organic fertilizer applied? (commercial name)')
# Fertilizer.dropna(subset=['FINAL_ANSWER'], inplace=True)
# Manure = Fertilizer.assign(manure = Fertilizer['FINAL_ANSWER'].str.contains('manure'))
# # Manure['manure'] = Manure['manure'].fillna(False).astype(bool)
# Manure['manure'] = Manure['manure'].astype(int)
# Frame_manure = Manure.drop(columns = ['QUESTION_DESC','FINAL_ANSWER'])


# ## creating a dataframe for pruning
# Pruning = ques.get_group('Do you prune?')
# Pruning.dropna(subset=['FINAL_ANSWER'], inplace=True)
# Pruning.loc[Pruning["FINAL_ANSWER"].str.contains('1', na=False), "FINAL_ANSWER"] = 1
# #Pruning.loc[(Pruning["FINAL_ANSWER"] == "1") or (Pruning["FINAL_ANSWER"] == '1 || 1') or (Pruning["FINAL_ANSWER"] == '0 || 1'), "FINAL_ANSWER"] = 1
# Pruning = Pruning.rename(columns={'FINAL_ANSWER':'prune'})
# # Pruning['prune'] = Pruning['prune'].fillna(0)
# Pruning['prune'] = Pruning['prune'].astype(int)
# Frame_prune = Pruning.drop(columns = ['QUESTION_DESC'])
# Frame_prune['prune'].describe()


"""creating a dataframe for weeding old version"""
# Weeding = ques.get_group('Level of weeding')
# Weeding.dropna(subset=['FINAL_ANSWER'], inplace=True)
# Weeding.loc[Weeding["FINAL_ANSWER"].str.contains('good', na=False), "FINAL_ANSWER"] = 2
# Weeding.loc[Weeding["FINAL_ANSWER"].str.contains('some', na=False), "FINAL_ANSWER"] = 1
# Weeding.loc[Weeding["FINAL_ANSWER"].str.contains('no', na=False), "FINAL_ANSWER"] = 0
# #Pruning.loc[(Pruning["FINAL_ANSWER"] == "1") or (Pruning["FINAL_ANSWER"] == '1 || 1') or (Pruning["FINAL_ANSWER"] == '0 || 1'), "FINAL_ANSWER"] = 1
# Weeding = Weeding.rename(columns={'FINAL_ANSWER':'weeding'})
# Frame_weeding = Weeding.drop(columns = ['QUESTION_DESC'])
# Frame_weeding['weeding'].describe()


"""creating a dataframe for mulching"""
practices = ques.get_group('What soil fertility practices can you see on the farm?')
practices.dropna(subset=['FINAL_ANSWER'], inplace=True)
mulching = practices.assign(mulching = practices['FINAL_ANSWER'].str.contains('Mulching'))
# mulching['mulching'] = mulching['mulching'].fillna(False).astype(bool)
mulching['mulching'] = mulching['mulching'].astype(int)
Frame_mulching = mulching.drop(columns = ['QUESTION_DESC','FINAL_ANSWER', "QUESTION_ID"])


## creating a dataframe for soil erosion control (yes= 1, no=0)
# practices = ques.get_group('Which soil erosion control practices did you apply on your cocoa plots in the last crop cycle?')
# practices.dropna(subset=['FINAL_ANSWER'], inplace=True)
# erosion = practices.assign(erosion_practices = practices['FINAL_ANSWER'].str.contains('r'))
# # erosion_practices['erosion_practices'] = erosion_practices['erosion_practices'].fillna(False).astype(bool)
# erosion['erosion_practices'] = erosion['erosion_practices'].astype(int)
# Frame_erosion = erosion.drop(columns = ['QUESTION_DESC','FINAL_ANSWER'])

def inter_quartiling(dfg, column):
    #country = dfg.iloc[1, 3]
    data = dfg#.loc[~(dfg[column] == 0.000000e+00)]
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75) 
    IQR = Q3 - Q1
  
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_outlied = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]   
    # plt.figure
    # sns.displot(df_outlied[column], bins=50,kde=False)
    # plt.title(f'distribution plot without outliers of {column} of {country}')
    # plt.show()
    return df_outlied

"""creating a dataframe for density"""
Density = ques.get_group('For the main plot, take 5 measurements of the distance between trees, and record them in the boxes below. The application will calculate an average for you.')
Density.dropna(subset=['FINAL_ANSWER'], inplace=True)
Frame_density = Density.rename(columns={'FINAL_ANSWER':'tree_density'})
Frame_density = Frame_density.drop(columns = ['QUESTION_DESC', "QUESTION_ID"])
Frame_density['tree_density'] = pd.to_numeric(Frame_density['tree_density'], errors='coerce')
Frame_density = Frame_density.groupby(by= 'COUNTRY_NAME', group_keys=False).apply(lambda dfg: inter_quartiling(dfg, 'tree_density'))
country = Frame_density.groupby(by='COUNTRY_NAME', group_keys=False)
for cntry in countries:
    df_country_density = country.get_group(cntry)
    plt.figure
    sns.displot(df_country_density['tree_density'], bins=30,kde=False)
    plt.title(f'distribution plot without outliers of {'tree_density'} of {cntry}')
    plt.show()
    print(cntry)
    print(df_country_density['tree_density'].describe())
    print(df_country_density['tree_density'].aggregate(['mean','median']))

# ## creating a dataframe for pruning practices
# prune_categories = ['Canopy shaping/shape pruning', 'Cutting branches to limit height (<3m)', 'Removal of  chupons', 'Removal of dead, infected, dry branches', 
#                     'Removal of dropping and hanging branches', 'Removal of infected pods (by pests & diseases)', 'Removal of interlocked branches between trees', 
#                     'Removal of interlocked branches inside tree', 'Removal of parasitic plants, mistletoes and epiphytes'] 

# #print(len(prune_categories))
# practices = ques.get_group('What pruning practices have you implemented on all of your cocoa plots in the last crop season? ENUMERATOR: Refer to guideline for the exact definition of each level.')
# practices.dropna(subset=['FINAL_ANSWER'], inplace=True)
# def convert_to_number(FINAL_ANSWER):
#     counter = 0
#     for x in prune_categories:
#         if x in FINAL_ANSWER
#:  
#             counter = counter + 1
#     return counter
# practices['n_prune_practices'] = practices['FINAL_ANSWER'].apply(convert_to_number)
# Frame_n_prune = practices.drop(columns = ['QUESTION_DESC', 'FINAL_ANSWER'])
# print(Frame_n_prune.shape)
# print(Frame_compost.shape)
# print(Frame_density.shape)
# print(Frame_erosion.shape)
# print(Frame_mulching.shape)
# print(Frame_prune.shape)
# print(Frame_weeding.shape)
# print(Frame_manure.shape)

# farmer_id = df_regression.groupby(by='olam_farmer_id').count()
# print(farmer_id)



# data_frames = [Frame_compost,  Frame_manure, Frame_prune, Frame_weeding, Frame_erosion, Frame_mulching, Frame_n_prune, Frame_density]
# df_merged = reduce(lambda  left,right: pd.merge(left,right, on= heading, how='outer'), data_frames)
# print(df_merged.head())
# print(df_merged.shape)
# df_merged.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables.csv', index=False)

"""creating weeding and pruning level variables, these are more subdivided variables than the previous made variables prune and weeding"""
prune_level = ques.get_group('What was the level of pruning on all of your cocoa plots in the last crop season?  ENUMERATOR: Refer to guideline for the exact definition of each level.')
prune_level.dropna(subset=['FINAL_ANSWER'], inplace=True)
prune_level.loc[prune_level["FINAL_ANSWER"].str.contains('Good', na=False), "FINAL_ANSWER"] = 3
prune_level.loc[prune_level["FINAL_ANSWER"].str.contains('Medium', na=False), "FINAL_ANSWER"] = 2
prune_level.loc[prune_level["FINAL_ANSWER"].str.contains('Poor', na=False), "FINAL_ANSWER"] = 1
prune_level.loc[prune_level["FINAL_ANSWER"].str.contains('No', na=False), "FINAL_ANSWER"] = 0
prune_level = prune_level.rename(columns={'FINAL_ANSWER':'prune_level'})
clean_prune = prune_level.drop(columns = ['QUESTION_DESC', "QUESTION_ID"])

weeding_level = ques.get_group('What was the level of weeding on all of your cocoa plots in the last crop cycle?  ENUMERATOR: Refer to guideline for the exact definition of each level.')
weeding_level.dropna(subset=['FINAL_ANSWER'], inplace=True)
weeding_level.loc[weeding_level["FINAL_ANSWER"].str.contains('good', na=False), "FINAL_ANSWER"] = 3
weeding_level.loc[weeding_level["FINAL_ANSWER"].str.contains('medium', na=False), "FINAL_ANSWER"] = 2
weeding_level.loc[weeding_level["FINAL_ANSWER"].str.contains('poor', na=False), "FINAL_ANSWER"] = 1
weeding_level.loc[weeding_level["FINAL_ANSWER"].str.contains('No', na=False), "FINAL_ANSWER"] = 0
weeding_level = weeding_level.rename(columns={'FINAL_ANSWER':'weeding_level'})
clean_weeding = weeding_level.drop(columns = ['QUESTION_DESC', "QUESTION_ID"])
print(clean_weeding['weeding_level'].value_counts())
print(clean_prune['prune_level'].value_counts())
all_data_frames = [Frame_mulching, Frame_density, clean_weeding, clean_prune]
all_df_merged = reduce(lambda  left,right: pd.merge(left,right, on= heading_2025, how='outer'), all_data_frames)
print(all_df_merged.head())
print(all_df_merged['mulching'].value_counts())
all_df_merged.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables+levels_2025.csv', index=False)

