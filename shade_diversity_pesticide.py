import pandas as pd

# df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')
# df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\tree_species_2024_question_id.csv')
df = pd.read_csv("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_concat_2025.csv")

pd.set_option('display.max_colwidth', None)
# heading = ['olam_farmer_id','Gender','country_name','region_name','district_name','place_name']
heading_2025 = ["FARMER_ID", "FARMER_GENDER", "FARM_ID", "REPORTING_YEAR", "COUNTRY_NAME", "REGION_NAME", "DISTRICT_NAME"]

ques = df.groupby(by='QUESTION_DESC')
# ques = df.groupby(by='question')
# fung = pd.read_excel('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx', sheet_name='Fung')
# herb = pd.read_excel('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx', sheet_name='Herb')
# insect = pd.read_excel('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx', sheet_name='Insect ')

import requests
import seaborn as sns
import matplotlib.pyplot as plt
q = ques.groups.keys()
df_QUESTION_DESC = ques.get_group('What species?')
print(df_QUESTION_DESC)
id = df_QUESTION_DESC.groupby(by= 'QUESTION_ID')
print(id.group_keys)
drop_df = id.get_group(7719)

"""question for 2025 does not work is about cocoa tree I believe"""
# tree_species = ['Acacia', 'Africa limba wood', 'Angolan mahogany', 'Avocado', 'Banana tree', 'Açaí', 'bitter kola', 'Afrormosia-Assamela', 'Awiemfosamina', 'Baku', 'Edinam', 'African Mahogany', 'Emire', 'Other', 'African mahogany', 'Antiaris', 'Oprono', 'Red cedar', 'Funtum', 'Lemon tree', 'Mango', 'Nyamedua', 'Guava', 'Long pepper tree', 'Afina', 'African teak', 'Akpi', 'Coffee', 'Oil palm', 'Orange', 'Papaya', 'Kyenkyen', 'Otie', 'bitter bush mango', 'Futum', 'Kapok', 'Rubber tree', 'Wild mango tree', 'African plum', 'African breadfruit', 'palm oil tree', 'other', 'Albizzia', 'Coconut palm', 'Teak', 'Oyina', 'Pouteria', 'Lemon', 'African ebony', 'African tulip tree', 'Plantain', 'silkrubber', 'Coconut', 'Utile', 'teck', 'Watapuo', 'African oil palm', 'Akye', 'Dwindwera', 'Gyapam', 'Kra bise', 'Odadee', 'Kwakuo bise', 'Oil Palm', 'African pear tree', 'African walnut', 'Atiokouo', 'Essia', 'Breadfruit', 'Pepper', 'Odwuma', 'red ironwood tree', 'Mahogany', 'Odum', 'African rosewood', 'Reddish brown timber', 'Plum tree', 'Ice cream bean', "Klaine's lovoa", 'Tangerine', 'bibolo wood', 'Doussi', 'Ingá', 'Pterocarpus', 'Ricinodendron heudelotii', 'Wenge', 'Balsa', 'Rambutan', 'Custard apple']
# drop_df = df.dropna(subset=['answer'])
# print(drop_df.head())
# def convert_to_number(answer):
#     counter = 0
#     for x in tree_species:
#         if x in answer:  
#             counter = counter + 1
#     return counter
# drop_df.loc[:,'n_tree_species'] = drop_df['answer'].apply(convert_to_number)
# print(drop_df['n_tree_species'].describe())
# print(drop_df['n_tree_species'].value_counts())

# print(species_clean.head(20))
"""creating number of species for 2025 data"""
df_question_drop = ques.get_group('How many different species of non-cocoa trees do you have in all your cocoa plots? (older than 2 years)')
print(df_question_drop.head())
df_question_drop = df_question_drop.copy()
# df_cote_clean = nan_cote.dropna(subset=['yield'])
df_question = df_question_drop.dropna(subset=['FINAL_ANSWER'])
df_question = df_question.copy()
df_question.loc[:, 'n_tree_species'] = pd.to_numeric(df_question['FINAL_ANSWER'], errors='coerce')
# pd.to_numeric(df_question['FINAL_ANSWER'], errors='coerce')
# question = df_question.rename(columns={'FINAL_ANSWER':'n_tree_species'})
df_final = df_question.drop(columns = ['QUESTION_DESC','FINAL_ANSWER', "QUESTION_ID"])
# sns.displot(df_final['n_tree_species'], bins=50,kde=False)
# plt.title(f'distribution plot of n_tree_species of all four countries')
# plt.show()
# print(df_final['n_tree_species'].value_counts())
df_final_year = df_final.loc[df_final['REPORTING_YEAR']==2025]

# # unkown = 'futum'
medicine_tree = ['Kra bise', 'Kwakuo bise', 'Nyamedua', 'Otie', 'Watapuo', 'bitter kola', 'Dwindwera', 'Kapok', 'Gyapam', 'African tulip tree', 'Neem']
N_fix_trees = ['Acacia', 'Pterocarpus', 'Antiaris', 'funtum', 'Ingá', 'Albizzia']
fruit_trees = ['Avocado', 'Açaí', 'Mango', 'Mangosteen', 'Guava', 'Coffee','Papaya','Coconut','African Breadfruit','Breadfruit', 'Coconut palm', 'Wild mango tree','bitter bush mango', 'African pear tree','African plum','Plum tree', 'Rambutan', 'Açaí', 'Pouteria', 'Ice cream bean', 'Pepper', 'Long pepper tree', 'Custard apple', 'African walnut',  'Akye', 'Odadee']
banana_trees = ['Plantain','Banana tree']
ioly_seed_trees = ['Akpi', 'Baku', 'Ricinodendron heudelotii']
timber_trees = ['African Mahogany',  'Afina', 'Angolan mahogany', 'Mahogany','African mahogany','Africa limba wood', 'African ebony', 'Doussi', 'Rubber tree','African rosewood','African teak', 'Teak', 'Reddish brown timber','bibolo wood', 'red ironwood tree', 'silkrubber', 'Balsa', 'Utile', 'Doussi', 'Red cedar', 'Afrormosia-Assamela', 'Wenge', 'Atiokouo', 'Awiemfosamina', 'Edinam', 'Emire', 'Essia', 'Funtum', "Klaine's lovoa", 'Kyenkyen', 'Odum', 'Odwuma', 'Oprono',  'Oyina', 'teck']
citrus_trees = ['Tangerine','Lemon', 'Lemon tree', 'Orange']
oil_palm_trees = ['Oil Palm', 'Oil palm', 'palm oil tree', 'African oil palm']
no_trees = ['none','None', 'other', 'Other']

"""creating categorical variable"""
tree_categories = [medicine_tree, N_fix_trees, fruit_trees, banana_trees, ioly_seed_trees, timber_trees, citrus_trees, oil_palm_trees]
tree_categories_names = ['medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']

drop_df = drop_df.copy()
def contains_species(answer):
    # counter = 0
    # for x in tree_species:
    #     if x in answer:  
    #         counter = counter + 1
    # return counter
    if isinstance(answer, str):  # Ensure 'FINAL_ANSWER' is a string to avoid errors with NaN values
        return int(sum(tree in answer for tree in category))
    return 0  # Default for missing values
for count in range (len(tree_categories)):
# Apply function to create 'timber_tr' column
    category = tree_categories[count]
    drop_df.loc[:, tree_categories_names[count]] = drop_df['FINAL_ANSWER'].apply(contains_species)
# print(drop_df.head(20))
n_fix_clean = drop_df.drop(columns = ['QUESTION_DESC', 'FINAL_ANSWER',  "QUESTION_ID"])
years = n_fix_clean.groupby(by='REPORTING_YEAR')
twenty_five = years.get_group(2025)
farms = twenty_five.groupby(['FARM_ID'], as_index=False).agg({"FARMER_ID": pd.Series.mode, "FARMER_GENDER": pd.Series.mode, "FARM_ID": pd.Series.mode, "REPORTING_YEAR": pd.Series.mode, "COUNTRY_NAME": pd.Series.mode, "REGION_NAME": pd.Series.mode, "DISTRICT_NAME": pd.Series.mode, 'medicine_tr':['sum'], 'N_fix_tr':['sum'], 'fruit_tr':['sum'], 'banana_tr':['sum'], 'ioly_seed_tr':['sum'], 'timber_tr':['sum'], 'citrus_tr':['sum'], 'oil_palm_tr':['sum']})
# farms = n_fix_clean.groupby(['olam_farmer_id'], as_index=False).agg({'farmer_id': pd.Series.mode,'Gender': pd.Series.mode,'country_name': pd.Series.mode,'region_name': pd.Series.mode,'district_name': pd.Series.mode,'place_name': pd.Series.mode, 'medicine_tr':['sum'], 'N_fix_tr':['sum'], 'fruit_tr':['sum'], 'banana_tr':['sum'], 'ioly_seed_tr':['sum'], 'timber_tr':['sum'], 'citrus_tr':['sum'], 'oil_palm_tr':['sum']})
farms.columns = farms.columns.droplevel(1)

# farms.columns = heading_2025 + tree_categories_names
# # diversity_sum = farms.sum()
print(farms.head(20))
df_merg = df_final_year.merge(farms, how= 'outer', left_on= heading_2025, right_on= heading_2025)
print(df_merg['N_fix_tr'].value_counts())
print(df_merg.head(20))


# df_tree = ques.get_group("How many trees of each species? (older than 2 years)")
# df_shade = df_tree.dropna(subset=['FINAL_ANSWER'])
# def summing(x):
#     # Split the string by " || " and convert each part to an integer
#     numbers = map(float, x.split(" || "))
#     # Sum the numbers and return the result
#     return sum(numbers)

# def inter_quartiling(dfg, important_columns):
#     #country = dfg.iloc[1, 3]
#     for column in important_columns:
#         if column == 'total_shade_trees':
#             data = dfg
#         else:
#             data = dfg.loc[~(dfg[column] == 0.000000e+00)]
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75) 
#         IQR = Q3 - Q1
    
#         # Define bounds
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df_outlied = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]   
#         dfg = df_outlied
#     return df_outlied

# df_shade['total_shade_trees'] = df_shade['FINAL_ANSWER'].apply(summing)
# df_final = df_shade.drop(columns = ['QUESTION_DESC', 'FINAL_ANSWER',  "QUESTION_ID"])

# df_final_2 = inter_quartiling(df_final, ['total_shade_trees']) 
# print(df_final_2['total_shade_trees'].describe())
# plt.figure
# sns.displot(df_final_2['total_shade_trees'], bins=50,kde=False)
# plt.title(f'distribution plot of total_shade_trees of all four countries')
# plt.show()

# numerical_QUESTION_DESCs = ['What is the total amount of fungicides applied to all your cocoa plots in the last crop cycle? (report in grams if non-liquid, and non-diluted liters if liquid)', 'What is the total amount of insecticide applied to all your cocoa plots in the last crop cycle?  (report in grams if non-liquid, and non-diluted liters if liquid)','What is the total amount of chemical herbicide applied to all your cocoa plots in the last crop cycle? If liquid, please answer the number of non-diluted liters of herbicide used.']
# num_variable_names = ['amount_fungicide','amount_insecticide','amount_herbicide']
# merging_dataframe = df.iloc[0:1]
# merging_dataframe = merging_dataframe.drop(columns = ['QUESTION_DESC','FINAL_ANSWER'])
# all_heading = heading_2025

# for count in range(3):
#     df_QUESTION_DESC = ques.get_group(numerical_QUESTION_DESCs[count])
#     # print(df_QUESTION_DESC.tail(20))
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC.dropna(subset=['FINAL_ANSWER'])
#     df_QUESTION_DESC_dropped['FINAL_ANSWER'] = df_QUESTION_DESC_dropped['FINAL_ANSWER'].apply(summing)
#     # print(df_QUESTION_DESC_dropped.tail(20))
#     # df_QUESTION_DESC_dropped['FINAL_ANSWER'] = pd.to_numeric(df_QUESTION_DESC_dropped['FINAL_ANSWER'], errors='coerce')
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.rename(columns={'FINAL_ANSWER': num_variable_names[count]})
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.drop(columns = ['QUESTION_DESC'])
#     # print("first")
#     # print(df_QUESTION_DESC_dropped)
#     # print("second")
#     # print(merging_dataframe)
#     dataframe = df_QUESTION_DESC_dropped.merge(merging_dataframe, how= 'outer', left_on= heading_2025, right_on= heading_2025)
#     merging_dataframe = dataframe
#     # all_heading.append(num_variable_names[count])
#     # print(all_heading)
#     # merging_dataframe = merging_dataframe.groupby(by= 'country_name', group_keys=False)[all_heading].apply(lambda dfg: inter_quartiling(dfg, num_variable_names[count]))
#     print(merging_dataframe[num_variable_names[count]].describe())

# merging_dataframe.loc[num_variable_names].fillna(0)
# merging_dataframe[num_variable_names] = merging_dataframe[num_variable_names].fillna(0)
# merging_dataframe['pesticide'] = merging_dataframe['amount_fungicide'] + merging_dataframe['amount_insecticide'] + merging_dataframe['amount_herbicide']
# df_question = ques.get_group('Total pesticides applied as active ingredients (a.i.)  kg/ha')
# df_question_dropped = df_question.dropna(subset=['FINAL_ANSWER'])
# df_question_dropped.loc[:,'pesticide'] = pd.to_numeric(df_question_dropped['FINAL_ANSWER'], errors='coerce')
# merging_dataframe = df_question_dropped.drop(columns = ['QUESTION_DESC', 'FINAL_ANSWER', "QUESTION_ID"])

# df_merg = df_final.merge(merging_dataframe, how= 'outer', left_on= heading_2025, right_on= heading_2025)
# df_shade_pesticide = df_merg.drop_duplicates(subset=['FARMER_ID'])
# # df_interquartiled = inter_quartiling(df_shade_pesticide, ['pesticide'])
# # print(df_interquartiled['pesticide'].describe())
# print(df_shade_pesticide.head())
# df_shade_pesticide.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_accurate_shade_2025.csv', index=False)

df_merg.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\tree_diversity_2025.csv', index=False)

# looking for certain words on a website
# request = requests.get('https://apps.worldagroforestry.org/suitable-tree/kenya?page=1', verify=False)
# for t in tree_species:
#     if (t in request.text):
#         print(t, "found")


"""Pesticide extended"""
# import openpyxl 

# wb = openpyxl.load_workbook("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx")
# print(wb.sheetnames)
# from pyxlsb import open_workbook

# def get_sheetnames_xlsb("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx"):
#   with open_workbook(filepath) as wb:
#      return wb.sheets
# import xlrd
# xls = xlrd.open_workbook(r'C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Copy of OFIS 2023 Multiple options Updated Format MCC.xlsx', on_demand=True)
# print(xls.sheet_names())
# sheets = [fung, herb, insect]
# # print(fung.head(20))

# numerical_QUESTION_DESCs = ['What is the total amount of fungicides applied to all your cocoa plots in the last crop cycle? (report in grams if non-liquid, and non-diluted liters if liquid)', 'What is the total amount of insecticide applied to all your cocoa plots in the last crop cycle?  (report in grams if non-liquid, and non-diluted liters if liquid)','What is the total amount of chemical herbicide applied to all your cocoa plots in the last crop cycle? If liquid, please answer the number of non-diluted liters of herbicide used.']
# amount_variable_names = ['amount_fungicide','amount_insecticide','amount_herbicide']
# merging_dataframe = df.iloc[0:1]
# merging_dataframe = merging_dataframe.drop(columns = ['question','answer'])
# all_heading = heading

# for count in range(3):
#     df_QUESTION_DESC = ques.get_group(numerical_QUESTION_DESCs[count])
#     # print(df_QUESTION_DESC.tail(20))
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC.dropna(subset=['answer'])
#     # print(df_QUESTION_DESC_dropped.tail(20))
#     # df_QUESTION_DESC_dropped['FINAL_ANSWER'] = pd.to_numeric(df_QUESTION_DESC_dropped['FINAL_ANSWER'], errors='coerce')
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.rename(columns={'answer': amount_variable_names[count]})
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.drop(columns = ['question'])
#     # print("first")
#     # print(df_QUESTION_DESC_dropped)
#     # print("second")
#     # print(merging_dataframe)
#     dataframe_amount = df_QUESTION_DESC_dropped.merge(merging_dataframe, how= 'outer', left_on= heading, right_on= heading)
#     merging_dataframe = dataframe_amount
    # all_heading.append(num_variable_names[count])
    # print(all_heading)
    # merging_dataframe = merging_dataframe.groupby(by= 'country_name', group_keys=False)[all_heading].apply(lambda dfg: inter_quartiling(dfg, num_variable_names[count]))
    # print(merging_dataframe[num_variable_names[count]].describe())

# dataframe_amount[amount_variable_names] = dataframe_amount[amount_variable_names].fillna(0)

# numerical_QUESTION_DESCs = ['Name of fungicide', 'Name of insecticide', 'Which chemical herbicides did you use in the last crop cycle?']
# num_variable_names = ['fungicide_name','insecticide_name','herbicide_name']
# merging_dataframe = df.iloc[0:1]
# merging_dataframe = merging_dataframe.drop(columns = ['question','answer'])
# all_heading = heading

# for count in range(3):
#     df_QUESTION_DESC = ques.get_group(numerical_QUESTION_DESCs[count])
#     # print(df_QUESTION_DESC.tail(20))
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC.dropna(subset=['answer'])
#     # print(df_QUESTION_DESC_dropped.tail(20))
#     # df_QUESTION_DESC_dropped['FINAL_ANSWER'] = pd.to_numeric(df_QUESTION_DESC_dropped['FINAL_ANSWER'], errors='coerce')
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.rename(columns={'answer': num_variable_names[count]})
#     df_QUESTION_DESC_dropped = df_QUESTION_DESC_dropped.drop(columns = ['question'])
#     # print("first")
#     # print(df_QUESTION_DESC_dropped)
#     # print("second")
#     # print(merging_dataframe)
#     dataframe_name = df_QUESTION_DESC_dropped.merge(merging_dataframe, how= 'outer', left_on= heading, right_on= heading)
#     merging_dataframe = dataframe_name
    # all_heading.append(num_variable_names[count])
    # print(all_heading)
    # merging_dataframe = merging_dataframe.groupby(by= 'country_name', group_keys=False)[all_heading].apply(lambda dfg: inter_quartiling(dfg, num_variable_names[count]))
    # print(merging_dataframe[num_variable_names[count]].describe())

# Example first dataframe (df_main)
# data = {'Fungicides': ['mancolin || OK-mil'],
#         'Amounts': ['300 || 400']}  # Corresponding amounts used

# # dataframe_name[num_variable_names] = dataframe_name[num_variable_names].fillna(0)
# df_main = dataframe_name.merge(dataframe_amount, how= 'outer', left_on= heading, right_on= heading)
# print(df_main.head())



# Example second dataframe (df_actives)
# active_substance_data = {'Fungicide': ['mancolin', 'OK-mil'],
#                         'Active_Substance': ['mancozeb', 'dimetomorph + oxyde de cuivre'],
#                         'Concentration': [800, 120]}  # Per kg
# df_actives = pd.DataFrame(active_substance_data)
# import math

# # 🔹 Function to compute total active substance
# def calculate_active_substance(row, count, df_actives):
#     # df_main.loc[:,num_variable_names[count]] = list(df_main[num_variable_names[count]])
#     if type(row[num_variable_names[count]]) == float or type(row[amount_variable_names[count]]) == float:
#         # if math.isnan(row[num_variable_names[count]]) or math.isnan(row[amount_variable_names[count]]):
#         return 0
#     # listed = list(row[num_variable_names[count]])
#     fungicides = row[num_variable_names[count]].split(' || ')  # Split fungicides
#     amounts = list(map(float, row[amount_variable_names[count]].split(' || ')))  # Split amounts & convert to int
#     # df[['Time']].apply(lambda x: x / 10000)
#     df_actives.loc[df_actives['Liter'] != 'L', 'active_unit'] = df_actives['active_unit']/1000
#     total_active_substance = 0
#     length = len(amounts)
#     if len(amounts) > len(fungicides):
#         length = len(fungicides)
#     for i in range(length):
#         concentration = df_actives.loc[df_actives['name'] == fungicides[i], 'active_unit'].values
#         if len(concentration) > 0:  # Ensure match exists
#             # print(i)
#             # print(amounts)
#             total_active_substance += amounts[i] * concentration[0]/1000  # Multiply amount by concentration
    
#     return total_active_substance

# # 🔹 Apply function to dataframe
# # df_main['Total_Active_Substance'] = df_main.apply(calculate_active_substance, axis=1)
# df_main['Total_fungicide'] = df_main.apply(calculate_active_substance, axis=1, count=0, df_actives=fung)
# df_main['Total_insecticide'] = df_main.apply(calculate_active_substance, axis=1, count=1, df_actives=insect)
# df_main['Total_herbicide'] = df_main.apply(calculate_active_substance, axis=1, count=2, df_actives=herb)
# df_main['pesticide'] = df_main['Total_fungicide'] + df_main['Total_herbicide'] + df_main['Total_insecticide']
# print(df_main)
# drop_columns = ['Total_fungicide', 'Total_insecticide', 'Total_herbicide', 'fungicide_name','insecticide_name','herbicide_name', 'amount_fungicide','amount_insecticide','amount_herbicide']
# dropping = df_main.drop(columns = drop_columns)
# df_merg = df_final.merge(dropping, how= 'outer', left_on= heading, right_on= heading)
# df_shade_pesticide = df_merg.drop_duplicates(subset=['olam_farmer_id'])
# # df_interquartiled = inter_quartiling(df_shade_pesticide, ['pesticide'])
# # print(df_interquartiled['pesticide'].describe())
# # print(df_shade_pesticide.head())
# df_shade_pesticide.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_accurate_shade.csv', index=False)

# print(df_shade_pesticide.head())
