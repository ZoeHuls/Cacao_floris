import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df = pd.read_csv("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_concat_2025.csv")

# df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')

# from functools import reduce
pd.set_option('display.max_colwidth', None)
# heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name']
# all_heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','QUESTION_DESC','FINAL_ANSWER']
heading_2025 = ["FARMER_ID", "FARMER_GENDER", "FARM_ID", "QUESTION_DESC_ID", "REPORTING_YEAR", "COUNTRY_NAME", "REGION_NAME", "DISTRICT_NAME"]

important_columns = ['farm_ha','dry_produced']
residual_managements = ['Burned', 'Exported off farm (for selling/cooking/animal feed)', 'Forced aeration compost', 'Incorporated or mulch evenly in the cocoa field','Left untreated in heaps (>50cm, knee level)', 'Left untreated in pits', 'Non-forced aeration compost']
residual_managements_names = ['Burned', 'Exported off farm (for selling/cooking/animal feed)', 'Forced aeration compost', 'Incorporated or mulch evenly in the cocoa field','Left untreated in heaps (>50cm, knee level)', 'Left untreated in pits', 'Non-forced aeration compost']

residual_managements_exclusive = ['Forced aeration compost', 'Incorporated or mulch evenly in the cocoa field', 'Left untreated in pits', 'Non-forced aeration compost']
ques = df.groupby(by='QUESTION_DESC')

df_management = ques.get_group('What did you do with open cocoa pods husks, branch residues, leaves and fallen cocoa trees and other post-harvest losses? (within a month of pod harvesting, prunning, tree remova, or fallen tree event)')
management = df_management.dropna(subset=['FINAL_ANSWER'])
print(management.head())
for count in range (len(residual_managements)):
    print(residual_managements[count])
    management = management.copy()
    # concatted_countries_unmerg.loc[:, 'distance'] = concatted_countries_unmerg['tree_density']
    management[residual_managements_names[count]] = management['FINAL_ANSWER'].str.contains(residual_managements[count])
    # Pruning.loc[Pruning["FINAL_ANSWER"].str.contains('1', na=False), "FINAL_ANSWER"] = 1
    #management.assign(xmanagement['FINAL_ANSWER'].str.contains(x)
    management[residual_managements_names[count]] = management[residual_managements[count]].astype(int)
management_clean = management.drop(columns = ['QUESTION_DESC', 'FINAL_ANSWER', 'QUESTION_ID'])
for x in residual_managements:
    print(management[x].value_counts(dropna = False))

print(management_clean.head())
print(management_clean.shape)
management_clean.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\residual_management_2025.csv', index=False)
