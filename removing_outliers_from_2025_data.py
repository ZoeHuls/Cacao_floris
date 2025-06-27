# setting up file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.diagnostic import het_breuschpagan

pd.set_option('display.max_columns', None)
heading_2025 = ["FARMER_ID", "FARMER_GENDER", "FARM_ID", "REPORTING_YEAR", "COUNTRY_NAME", "REGION_NAME", "DISTRICT_NAME"]

df_yield = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\database_yield_compressed_2025.csv')
#df_practice = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables.csv')
df_age = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\percentage_age_compressed_2025.csv')
df_shade_pesticide = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_accurate_shade_2025.csv')
# variables are already interquartiled
df_control = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_variables_without_irrigation_2025.csv')
df_practice_levels = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables+levels_2025.csv')
df_residual_management = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\residual_management_2025.csv') 
df_tree_diversity = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\tree_diversity_2025.csv')
df_org_chem_fertilizer = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\chemical_org_fertilizer_2025.csv')
## pest_disease, education, surrounding and irrigation is not included
# df_pest_disease = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Pest&disease.csv') 
#  df_more_control = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\surrounding&education.csv')
df_irrigation = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_irrigation_2025.csv')

df_read = [df_yield, df_age, df_control, df_practice_levels, df_residual_management, df_tree_diversity, df_shade_pesticide, df_org_chem_fertilizer, df_irrigation]
count = 0
for framed in df_read:
    frame = framed.drop_duplicates(subset=['FARMER_ID'])
    grp_frame = frame.groupby(by='COUNTRY_NAME', group_keys=False)
    cote_grp = grp_frame.get_group("COTE D'IVOIRE")
    ghana_grp = grp_frame.get_group("GHANA")
    nigeria_grp = grp_frame.get_group("NIGERIA")
    cameroon_grp = grp_frame.get_group("CAMEROON")
    if count == 0:
        cote = cote_grp
        ghana = ghana_grp
        nigeria = nigeria_grp
        cameroon = cameroon_grp
        count = 1
    else:
        cote = cote_grp.merge(cote, how = 'outer', left_on= heading_2025, right_on= heading_2025)
        ghana = ghana_grp.merge(ghana, how = 'outer', left_on= heading_2025, right_on= heading_2025)
        nigeria = nigeria_grp.merge(nigeria, how = 'outer', left_on= heading_2025, right_on= heading_2025)
        cameroon = cameroon_grp.merge(cameroon, how = 'outer', left_on= heading_2025, right_on= heading_2025)

countries_df = [cote, ghana, nigeria, cameroon]
df_suitability = pd.read_excel("C:\\Users\\zoe.huls\\Cacao_floris\\Suitability index table 2.xlsx")

important_columns = ['farm_ha','dry_produced','total_shade_trees']
columns_start_0 = ['total_shade_trees', 'shade/ha']
countries_df = [cote, ghana, nigeria, cameroon]
cote_main = pd.DataFrame
ghana_main = pd.DataFrame 
nigeria_main = pd.DataFrame
cameroon_main = pd.DataFrame
countries_main = [cote_main, ghana_main, nigeria_main, cameroon_main]
cote_yield = pd.DataFrame
ghana_yield = pd.DataFrame 
nigeria_yield = pd.DataFrame
cameroon_yield = pd.DataFrame
countries_yield = [cote_yield, ghana_yield, nigeria_yield, cameroon_yield]

"""exploratory analysis function print: average yield per category"""
def exploratory_analysis (df, columns):
    for column in columns:          
        plt.figure()
        sns.boxplot(x=column, y='yield', data=df)
        plt.title(f'Box Plot without Outliers of {column} of {df.iloc[1, 2]}')
        plt.show()
        print(df[column].value_counts(dropna = False))

def inter_quartiling(dfg, important_columns):
    #country = dfg.iloc[1, 3]
    for column in important_columns:
        if column in columns_start_0:
            data = dfg
        else:
            data = dfg.loc[~(dfg[column] == 0.000000e+00)]
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75) 
        IQR = Q3 - Q1
    
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outlied = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]   
        dfg = df_outlied
    df_outlied = df_outlied.loc[df_outlied['REPORTING_YEAR']==2025]
    return df_outlied

"""removing outliers from important columns"""
for column in important_columns:
    for counter in range (4):
        # print(countries_df[counter]['n_tree_species'].value_counts())
        countries_main[counter] = inter_quartiling(countries_df[counter], important_columns) 


"""Making farm/ha variable"""     
for countries in countries_main:
    countries['yield'] = countries['dry_produced']/countries['farm_ha']
    countries['shade/ha'] = countries['total_shade_trees']/countries['farm_ha']
for counter in range (4):   
    countries_yield[counter] = inter_quartiling(countries_main[counter], ['yield', 'shade/ha']) 


def exploratory_analysis_numerical (df, columns):
    for column in columns:
        sns.displot(df[column], bins=50,kde=False)
        plt.title(f'distribution plot without of {column} of {df.iloc[1, 2]}')
        plt.show()
        print(df[column].describe())
        df_regress = df.dropna(subset= [column, 'yield'])
        y = df_regress['yield']
        X = df_regress[[column]]
        X['Constant'] = 1
        m1 = sm.OLS(y, X).fit()
        print(m1.summary())

columns = ['n_tree_species', 'Burned', 'Exported off farm (for selling/cooking/animal feed)', 'Forced aeration compost', 'Incorporated or mulch evenly in the cocoa field', 'Left untreated in heaps (>50cm, knee level)', 'Left untreated in pits', 'Non-forced aeration compost', 'mulching', 'weeding_level', 'prune_level', 'pest_level', 'disease_level']

# for countries in countries_yield:
#     exploratory_analysis(countries, columns)

"""Cote d'Ivoire"""
drop_columns = [] #['amount_fungicide', 'amount_insecticide', 'amount_herbicide','many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']
ivoire_2 = countries_yield[0]
cote_small = ivoire_2.drop(columns= drop_columns)
df_cote = cote_small.drop_duplicates(subset=['FARMER_ID'])
all_country_variables = ['FARMER_ID', 'REGION_NAME', 'yield', 'tree_density', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha', 'pesticide']

nan_cote = df_cote.dropna(subset= all_country_variables)
df_cote_clean = nan_cote.dropna(subset=['yield'])

low_values = df_cote_clean.loc[~(df_cote_clean['yield'] < 15)]
cote_values = low_values.loc[(low_values['n_tree_species']  < 5) & (low_values['pesticide'] < 18390.87)]


"""Ghana"""
drop_columns = [] #['many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']
ghana_2 = countries_yield[1]

ghana_small = ghana_2.drop(columns= drop_columns)
df_ghana = ghana_small.drop_duplicates(subset=['FARMER_ID'])

nan_ghana = df_ghana.dropna(subset= all_country_variables)
df_ghana_clean = nan_ghana.dropna(subset=['yield'])

low_values = df_ghana_clean.loc[~(df_ghana_clean['yield'] < 20)]
values_ghana = low_values.loc[(low_values['n_tree_species']  < 5) & (low_values['pesticide'] < 57692.32)]

"""Nigeria"""
drop_columns = [] #['many_plots', 'n_plots_>4ha', 'n_hh_members_depend_income', 'wet_produced', 'farm_ha', 'n_prune_practices']
nigeria_2 = countries_yield[2]
nigeria_small = nigeria_2.drop(columns= drop_columns)
df_nigeria = nigeria_small.drop_duplicates(subset=['FARMER_ID'])

nan_nigeria = df_nigeria.dropna(subset= all_country_variables)
df_nigeria_clean = nan_nigeria.dropna(subset=['yield'])
values_nigeria = df_nigeria_clean.loc[(df_nigeria_clean['n_tree_species'] < 5) & (df_nigeria_clean['pesticide'] < 1.638153e+06)]



"""cameroon"""
cameroon_2 = countries_yield[3]
drop_columns = [] #['many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']

cameroon_small = cameroon_2.drop(columns= drop_columns)
df_cameroon = cameroon_small.drop_duplicates(subset=['FARMER_ID'])

nan_cameroon = df_cameroon.dropna(subset= all_country_variables)
df_cameroon_clean = nan_cameroon.dropna(subset=['yield'])

low_values = df_cameroon_clean.loc[~(df_cameroon_clean['yield'] < 20)]
n_treee_cameroon = low_values.loc[(low_values['n_tree_species'] < 5) & (low_values['pesticide'] < 143178.96)]


# # Making a box plots of the different countries yield and tree density for comparison
# combined_yields = pd.DataFrame({'Cote d"Ivoire': cote_values['yield'],
#                                 'ghana': values_ghana['yield'],
#                                 'Nigeria': values_nigeria['yield'],
#                                 'cameroon': n_treee_cameroon['yield']})

# sns.set_style('white')
# sns.boxplot(data=combined_yields, palette='flare')
# plt.title(f'Box Plot of yield of the different countries')
# sns.despine()
# plt.show()

"""creating binairy variable for country"""
cote_bin = ['Ghana','Cameroon','Nigeria']
ghana_bin = ['Cameroon','Nigeria', "Cote_d'Ivoire"]
cameroon_bin = ['Ghana','Nigeria', "Cote_d'Ivoire"]
nigeria_bin = ['Ghana','Cameroon', "Cote_d'Ivoire"]

cote_values.loc[:, "Cote_d'Ivoire"] = 1
cote_values.loc[:, cote_bin] = 0
values_nigeria = values_nigeria.copy()
values_nigeria.loc[:, 'Nigeria'] = 1
values_nigeria.loc[:, nigeria_bin] = 0
values_ghana.loc[:, 'Ghana'] = 1
values_ghana.loc[:, ghana_bin] = 0
n_treee_cameroon = n_treee_cameroon.copy()
n_treee_cameroon.loc[:,'Cameroon'] = 1
n_treee_cameroon.loc[:, cameroon_bin] = 0

list_all_variables = ["REPORTING_YEAR", 'FARMER_ID', 'REGION_NAME', 'Irrigation_amount', 'COUNTRY_NAME', 'yield', 'fertilizer', 'farm_ha','tree_density', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 5-20', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'Nigeria', 'shade/ha', 'pesticide', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
all_countries = [cote_values[list_all_variables], values_nigeria[list_all_variables], values_ghana[list_all_variables], n_treee_cameroon[list_all_variables]]
concatted_countries_unmerg = pd.concat((all_countries), axis=0, ignore_index=True)
# exploratory_analysis(concatted_countries_unmerg, ['n_tree_species'])

"""IMPORTANT treedistance --> treedensity """
concatted_countries_unmerg = concatted_countries_unmerg.copy()
concatted_countries_unmerg.loc[:, 'distance'] = concatted_countries_unmerg['tree_density']
concatted_countries_unmerg.loc[:, 'tree_density'] = concatted_countries_unmerg['tree_density'].apply(lambda x: 10000/(x**2))
# concatted_countries = concatted_countries_unmerg.groupby(by='REGION_NAME')
# keys_concatted = concatted_countries.groups.keys()


"""Creating irrigation variable"""
# concatted_countries.fillna({'Irrigation_amount': 0}, inplace=True)
# concatted_countries = concatted_countries.loc[~(concatted_countries['Irrigation_amount'] > 0)]

"""Making numerical suitability index"""
df_suitability_drop = df_suitability.drop(columns = 'map_name')
concatted_countries = pd.merge(concatted_countries_unmerg, df_suitability_drop, how="left", right_on="region_name", left_on="REGION_NAME")
concatted_countries = concatted_countries.rename(columns={"mean": "suitability_index"})

"""linear interpolation rescaling the variable"""
# concatted_countries.loc[:, 'tree_density'] = concatted_countries['tree_density'].apply(lambda x: (x-346.133045) * (5 - 0) / (2868.873270 - 346.133045))

"""Creating irrigation variable"""
concatted_countries.fillna({'Irrigation_amount': 0}, inplace=True)
concatted_countries = concatted_countries.loc[~(concatted_countries['Irrigation_amount'] > 0)]


"""Creating fertilizer variable"""
concatted_countries.fillna({'fertilizer': 0}, inplace=True)
concatted_countries.loc[:, 'fertilizer'] = (concatted_countries['fertilizer'])/concatted_countries['farm_ha']

concatted_countries = concatted_countries.loc[~(concatted_countries['fertilizer'] > 306.122)]

concatted_countries['pesticide'] =  concatted_countries['pesticide']/1000

# concatted_countries = inter_quartiling(concatted_countries, ['pesticide'])
# concatted_countries = inter_quartiling(concatted_countries, ['pesticide'])
# print(concatted_countries['pesticide'].describe())
# concatted_countries = concatted_countries.loc[concatted_countries['pesticide'] < 25000]
# plt.figure
# sns.displot(concatted_countries['pesticide'], bins=50,kde=False)
# plt.title(f'distribution plot of tree density of all countries')
# plt.show()
# concatted_countries = concatted_countries.loc[~(concatted_countries['pesticide'] < 2)]

concatted_countries = concatted_countries.loc[~(concatted_countries['yield'] < 20)]
concatted_countries = inter_quartiling(concatted_countries, ['yield'])

# plt.figure
# sns.displot(concatted_countries['yield'], bins=50,kde=False)
# plt.title(f'distribution plot of yield of all countries')
# plt.show()
# exploratory_analysis(concatted_countries, ['n_tree_species'])

"""Creating interaction variables"""
concatted_countries['average_age'] = (concatted_countries['%_age 0-5']* 2.5 + concatted_countries['%_age 5-20']*13 + concatted_countries['%_age 20-25']*22.5 + concatted_countries['%_age >25']*32)/100
concatted_countries['tree_density*average_age'] = concatted_countries['tree_density']*concatted_countries['average_age']
concatted_countries['tree_density^2'] = concatted_countries['tree_density']**2
concatted_countries['tree_density^3'] = concatted_countries['tree_density']**3
concatted_countries['tree_density*n_tree_species'] = concatted_countries['tree_density']*concatted_countries['n_tree_species']
concatted_countries['tree_density*%_age 0-5'] = concatted_countries['tree_density']*concatted_countries['%_age 0-5']
concatted_countries['tree_density*%_age 20-25'] = concatted_countries['tree_density']*concatted_countries['%_age 20-25']
concatted_countries['tree_density*%_age >25'] = concatted_countries['tree_density']*concatted_countries['%_age >25']
concatted_countries['tree_density*disease_level'] = concatted_countries['tree_density']*concatted_countries['disease_level']
concatted_countries['tree_density*pest_level'] = concatted_countries['tree_density']*concatted_countries['pest_level']
concatted_countries['tree_density*shade/ha'] = concatted_countries['tree_density']*concatted_countries['shade/ha']
print(concatted_countries.head(20))
concatted_countries.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_unaccerate_2025_all.csv', index=False)


# y = concatted_countries[['yield']]

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'pest_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
# X['Constant'] = 1
# m2 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m2.summary())

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'suitability_index','pesticide', 'weeding_level', 'prune_level', 'pest_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
# X['Constant'] = 1
# m2 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m2.summary())
# """Testing for normal distribution"""
# import scipy.stats as stats
# # create Q-Q plot with 45-degree line added to plot
# # Compute residuals
# residuals = m2.resid  

# Plot histogram
# plt.hist(residuals, bins=50, edgecolor='black', density=True)
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Histogram of Residuals")
# plt.grid()
# plt.show()

# # Q-Q plot
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid()
# plt.show()

# """Correlation matrix"""
# # # multi_c_df = concatted_countries[['tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Cameroon_low', 'Ghana_low', 'Nigeria_low', 'Cote_divoire_low', 'Ghana_medium', 'Nigeria_medium', 'Cameroon_medium', 'Cote_divoire_medium', 'Ghana_high', 'Cameroon_high', 'Cote_divoire_high', 'shade/ha']]
# multi_c_df = concatted_countries[['suitability_index', 'tree_density', 'weeding_level', 'prune_level', 'disease_level', 'fertilizer', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'pesticide', 'shade/ha', 'Cameroon', 'Ghana', 'Nigeria', "Cote_d'Ivoire"]]

# correlation_matrix = multi_c_df.corr()
# plt.figure(figsize=(10, 6))
# # Create a heatmap for the correlation matrix
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# # Title for the heatmap
# plt.title("Correlation Heatmap", fontsize=16)
# # Show the heatmap
# plt.show()

