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

heading = ['olam_farmer_id','Gender','country_name','region_name','district_name','place_name']
countries = ["COTE D'IVOIRE",'GHANA', "NIGERIA","CAMEROON"]

df_yield = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\database_yield_compressed.csv')
#df_practice = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables.csv')
df_age = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\percentage_age_compressed.csv')
df_shade_pesticide = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_accurate_shade.csv')
# variables are already interquartiled
df_control = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_variables_without_irrigation.csv')
df_practice_levels = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\practice_variables+levels.csv')
df_residual_management = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\residual_management.csv') 
df_tree_diversity = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\tree_diversity.csv')

df_org_chem_fertilizer = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\chemical_org_fertilizer.csv')
## pest_disease, education, surrounding and irrigation is not included
# df_pest_disease = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Pest&disease.csv') 
#  df_more_control = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\surrounding&education.csv')
df_irrigation = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\control_irrigation.csv')

df_read = [df_yield, df_age, df_control, df_practice_levels, df_residual_management, df_tree_diversity, df_shade_pesticide, df_org_chem_fertilizer, df_irrigation]
count = 0
for framed in df_read:
    frame = framed.drop_duplicates(subset=['olam_farmer_id'])
    if "farmer_id" in frame.columns: 
        frame = frame.drop(columns = ['farmer_id'])
    grp_frame = frame.groupby(by='country_name', group_keys=False)
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
        cote = cote_grp.merge(cote, how = 'outer', left_on= heading, right_on= heading)
        ghana = ghana_grp.merge(ghana, how = 'outer', left_on= heading, right_on= heading)
        nigeria = nigeria_grp.merge(nigeria, how = 'outer', left_on= heading, right_on= heading)
        cameroon = cameroon_grp.merge(cameroon, how = 'outer', left_on= heading, right_on= heading)

countries_df = [cote, ghana, nigeria, cameroon]

df_suitability = pd.read_excel("C:\\Users\\zoe.huls\\Cacao_floris\\Suitability index table 2.xlsx")

import matplotlib.colors
concatted_countries_2025 = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\pesticide_unaccerate_2025_all.csv')
concatted_countries_2025 = concatted_countries_2025.drop(columns=['FARMER_ID', 'REPORTING_YEAR', 'REGION_NAME', 'COUNTRY_NAME'])
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
        plt.title(f'Box Plot of all the different African countries of {column}')
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
    return df_outlied

"""removing outliers from important columns"""
for column in important_columns:
    for counter in range (4):
        countries_main[counter] = inter_quartiling(countries_df[counter], important_columns) 

"""Making farm/ha variable"""     
for countries in countries_main:
    countries['yield'] = countries['dry_produced']/countries['farm_ha']
    countries['shade/ha'] = countries['total_shade_trees']/countries['farm_ha']
    # print(countries['pesticide'].describe())
for counter in range (4):   
    countries_yield[counter] = inter_quartiling(countries_main[counter], ['yield', 'shade/ha']) 
    # print(countries_yield[counter]['shade/ha'].describe())
def exploratory_analysis_numerical (df, columns):
    for column in columns:
        sns.displot(df[column], bins=50,kde=False)
        plt.title(f'distribution plot without of {column} of {df.iloc[1, 2]}')
        plt.show()
        print(df[column].describe())
        # df_regress = df.dropna(subset= [column, 'yield'])
        # y = df_regress['yield']
        # X = df_regress[[column]]
        # X['Constant'] = 1
        # m1 = sm.OLS(y, X).fit()
        # print(m1.summary())

"""Cote d'Ivoire"""
drop_columns = [] #['amount_fungicide', 'amount_insecticide', 'amount_herbicide','many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']
ivoire_2 = countries_yield[0]
cote_small = ivoire_2.drop(columns= drop_columns)
df_cote = cote_small.drop_duplicates(subset=['olam_farmer_id'])
all_country_variables = ['olam_farmer_id', 'region_name', 'yield', 'tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha', 'pesticide', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']

nan_cote = df_cote.dropna(subset= all_country_variables)
df_cote_clean = nan_cote.dropna(subset=['yield'])

low_values = df_cote_clean.loc[~(df_cote_clean['yield'] < 15)]
cote_values = low_values.loc[(low_values['n_tree_species']  < 5) & (low_values['pesticide'] < 2.03)]


print(low_values['timber_tr'].value_counts())
"""Ghana """
drop_columns = [] #['many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']
ghana_2 = countries_yield[1]

ghana_small = ghana_2.drop(columns= drop_columns)
df_ghana = ghana_small.drop_duplicates(subset=['olam_farmer_id'])

nan_ghana = df_ghana.dropna(subset= all_country_variables)
df_ghana_clean = nan_ghana.dropna(subset=['yield'])

low_values = df_ghana_clean.loc[~(df_ghana_clean['yield'] < 20)]
values_ghana = low_values.loc[(low_values['n_tree_species']  < 5) & (low_values['pesticide'] < 13.07)]

"""Nigeria"""
drop_columns = [] #['many_plots', 'n_plots_>4ha', 'n_hh_members_depend_income', 'wet_produced', 'farm_ha', 'n_prune_practices']
nigeria_2 = countries_yield[2]
nigeria_small = nigeria_2.drop(columns= drop_columns)
df_nigeria = nigeria_small.drop_duplicates(subset=['olam_farmer_id'])

nan_nigeria = df_nigeria.dropna(subset= all_country_variables)
df_nigeria_clean = nan_nigeria.dropna(subset=['yield'])

values_nigeria = df_nigeria_clean.loc[(df_nigeria_clean['n_tree_species'] < 5) & (df_nigeria_clean['pesticide'] < 232.35)]

"""cameroon"""
cameroon_2 = countries_yield[3]
drop_columns = [] #['many_plots','n_plots_>4ha','n_hh_members_depend_income', 'wet_produced', 'farm_ha','n_prune_practices']

cameroon_small = cameroon_2.drop(columns= drop_columns)
df_cameroon = cameroon_small.drop_duplicates(subset=['olam_farmer_id'])

nan_cameroon = df_cameroon.dropna(subset= all_country_variables)
df_cameroon_clean = nan_cameroon.dropna(subset=['yield'])

low_values = df_cameroon_clean.loc[~(df_cameroon_clean['yield'] < 20)]
n_treee_cameroon = low_values.loc[(low_values['n_tree_species']  < 5) & (low_values['pesticide'] < 3.27)]

"""printing a scatterplot with three variables of tree density age and yield"""
# fig, ax = plt.subplots( figsize = (9,6))
# scatter = ax.scatter(low_values['tree_density'], Y, c = low_values['%_age 0-5'] , cmap = 'Greens' , s = 80)
# legend = ax.legend(*scatter.legend_elements(), loc="lower right",
#                    title="%_age 0-5")
# ax.add_artist(legend)
# ax.set_xlabel('tree_density')
# ax.set_ylabel('Yield')
# ax.set_title('Scatter chart, Cameroon showing the interaction between tree density and %_age 0-5', size = 14)
# plt.show()

"""All countries together again"""
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

list_all_variables = ['olam_farmer_id', 'region_name', 'Irrigation_amount', 'country_name', 'yield', 'chem_fertilizer', 'org_fertilizer', 'farm_ha','tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 5-20', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'Nigeria', 'shade/ha', 'pesticide', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
all_countries = [cote_values[list_all_variables], values_nigeria[list_all_variables], values_ghana[list_all_variables], n_treee_cameroon[list_all_variables]]

"""compare cote d'ivoire with other countries"""
# all_countries = [values_nigeria[list_all_variables], values_ghana[list_all_variables], n_treee_cameroon[list_all_variables]]
# all_countries = [cote_values[list_all_variables]]

concatted_countries_unmerg = pd.concat((all_countries), axis=0, ignore_index=True)

"""IMPORTANT treedistance --> treedensity """
concatted_countries_unmerg = concatted_countries_unmerg.copy()
concatted_countries_unmerg.loc[:, 'distance'] = concatted_countries_unmerg['tree_density']
concatted_countries_unmerg.loc[:, 'tree_density'] = concatted_countries_unmerg['tree_density'].apply(lambda x: 10000/(x**2))


"""Making numerical suitability index"""
df_suitability_drop = df_suitability.drop(columns = 'map_name')
concatted_countries = pd.merge(concatted_countries_unmerg, df_suitability_drop, how="left", on="region_name")
concatted_countries = concatted_countries.rename(columns={"mean": "suitability_index"})

"""linear interpolation rescaling the variable"""
# concatted_countries.loc[:, 'tree_density'] = concatted_countries['tree_density'].apply(lambda x: (x-346.133045) * (5 - 0) / (2868.873270 - 346.133045))

"""Creating irrigation variable"""
concatted_countries.fillna({'Irrigation_amount': 0}, inplace=True)
concatted_countries = concatted_countries.loc[~(concatted_countries['Irrigation_amount'] > 0)]


"""Creating fertilizer variable"""
concatted_countries.fillna({'org_fertilizer': 0}, inplace=True)
concatted_countries.fillna({'chem_fertilizer': 0}, inplace=True)

concatted_countries.loc[:, 'fertilizer'] = (concatted_countries['org_fertilizer'] + concatted_countries['chem_fertilizer'])/concatted_countries['farm_ha']

concatted_countries = concatted_countries.loc[~(concatted_countries['fertilizer'] > 579)]
# print(concatted_countries['fertilizer'].describe())

concatted_countries['pesticide'] =  concatted_countries['pesticide']/concatted_countries['farm_ha']
# concatted_countries = concatted_countries.loc[~(concatted_countries['pesticide'] > 100)]
# concatted_countries = concatted_countries.loc[~(concatted_countries['pesticide'] > 360)]
concatted_countries = concatted_countries.loc[~(concatted_countries['yield'] < 20)]
concatted_countries = inter_quartiling(concatted_countries, ['yield'])

"""distribution plot tree density or other"""
# plt.figure
# sns.displot(concatted_countries['pesticide'], bins=50,kde=False)
# plt.title(f'distribution plot of tree density of all countries')
# plt.show()
# print(concatted_countries['pesticide'].describe())
# """exploratory analysis"""
# explore_columns = ['weeding_level', 'suitability_index','prune_level', 'disease_level', 'pest_level', 'n_tree_species', 'Incorporated or mulch evenly in the cocoa field', 'Nigeria', 'Ghana', 'Cameroon', "Cote_d'Ivoire"]
# exploratory_analysis(concatted_countries, ['n_tree_species'])

"""Creating interaction variables"""
concatted_countries['average_age'] = (concatted_countries['%_age 0-5']* 2.5 + concatted_countries['%_age 5-20']*13 + concatted_countries['%_age 20-25']*22.5 + concatted_countries['%_age >25']*32)/100
concatted_countries['tree_density*average_age'] = concatted_countries['tree_density']*concatted_countries['average_age']

concatted_countries['tree_density^3'] = concatted_countries['tree_density']**3

concatted_countries = concatted_countries.drop(columns='olam_farmer_id')
concat_years = [concatted_countries, concatted_countries_2025]
concatted_countries = pd.concat(concat_years, axis=0, ignore_index=True)
# concatted_countries = concatted_countries.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'AKURE 1 AKWA IBOM 1 ASHANTI 1 BRONG AHAFO 3 CENTRAL REGION 1 CENTRE 1 CENTRE REGION 1 CENTRE-EST 1 CENTRE-OUEST 1 CROSS RIVER 1 EAST 1 EASTERN REGION 1 EDO 1 EKITI 1 EST 1 IDANRE 1 IKOM 1 Littoral 1 NORD 1 OGUN 1 ONDO 1 OSUN 1 OUEST 1 OYO 1 SOUTH 1 SOUTH WEST 1 SUD 1 SUD-EST 1 SUD-OUEST 1 TARABA 1 WESTERN 1 WESTERN NORTH 1 West Region 1'])
# print(concatted_countries['tree_density'].describe())
concatted_countries['tree_density'] = np.log(concatted_countries['tree_density'])
# concatted_countries['shade/ha'] = np.log(concatted_countries['shade/ha'])
concatted_countries['tree_density^2'] = concatted_countries['tree_density']**2
concatted_countries['tree_density*n_tree_species'] = concatted_countries['tree_density']*concatted_countries['n_tree_species']
concatted_countries['tree_density*%_age 0-5'] = concatted_countries['tree_density']*concatted_countries['%_age 0-5']
concatted_countries['tree_density*%_age 20-25'] = concatted_countries['tree_density']*concatted_countries['%_age 20-25']
concatted_countries['tree_density*%_age >25'] = concatted_countries['tree_density']*concatted_countries['%_age >25']
concatted_countries['tree_density*disease_level'] = concatted_countries['tree_density']*concatted_countries['disease_level']
concatted_countries['tree_density*pest_level'] = concatted_countries['tree_density']*concatted_countries['pest_level']
concatted_countries['tree_density*shade/ha'] = concatted_countries['tree_density']*concatted_countries['shade/ha']
concatted_countries['tree_density*medicine_tr'] = concatted_countries['tree_density']*concatted_countries['medicine_tr']
concatted_countries['tree_density*N_fix_tr'] = concatted_countries['tree_density']*concatted_countries['N_fix_tr']
concatted_countries['tree_density*fruit_tr'] = concatted_countries['tree_density']*concatted_countries['fruit_tr']
concatted_countries['tree_density*banana_tr'] = concatted_countries['tree_density']*concatted_countries['banana_tr']
concatted_countries['tree_density*ioly_seed_tr'] = concatted_countries['tree_density']*concatted_countries['ioly_seed_tr']
concatted_countries['tree_density*timber_tr'] = concatted_countries['tree_density']*concatted_countries['timber_tr']
concatted_countries['tree_density*citrus_tr'] = concatted_countries['tree_density']*concatted_countries['citrus_tr']
concatted_countries['tree_density*oil_palm_tr'] = concatted_countries['tree_density']*concatted_countries['oil_palm_tr']

concatted_countries['tree_density^2*n_tree_species'] = concatted_countries['tree_density^2']*concatted_countries['n_tree_species']
concatted_countries['tree_density^2*%_age 0-5'] = concatted_countries['tree_density^2']*concatted_countries['%_age 0-5']
concatted_countries['tree_density^2*%_age 20-25'] = concatted_countries['tree_density^2']*concatted_countries['%_age 20-25']
concatted_countries['tree_density^2*%_age >25'] = concatted_countries['tree_density^2']*concatted_countries['%_age >25']
concatted_countries['tree_density^2*disease_level'] = concatted_countries['tree_density^2']*concatted_countries['disease_level']
concatted_countries['tree_density^2*pest_level'] = concatted_countries['tree_density^2']*concatted_countries['pest_level']
concatted_countries['tree_density^2*shade/ha'] = concatted_countries['tree_density^2']*concatted_countries['shade/ha']
concatted_countries['tree_density^2*medicine_tr'] = concatted_countries['tree_density^2']*concatted_countries['medicine_tr']
concatted_countries['tree_density^2*N_fix_tr'] = concatted_countries['tree_density^2']*concatted_countries['N_fix_tr']
concatted_countries['tree_density^2*fruit_tr'] = concatted_countries['tree_density^2']*concatted_countries['fruit_tr']
concatted_countries['tree_density^2*banana_tr'] = concatted_countries['tree_density^2']*concatted_countries['banana_tr']
concatted_countries['tree_density^2*ioly_seed_tr'] = concatted_countries['tree_density^2']*concatted_countries['ioly_seed_tr']
concatted_countries['tree_density^2*timber_tr'] = concatted_countries['tree_density^2']*concatted_countries['timber_tr']
concatted_countries['tree_density^2*citrus_tr'] = concatted_countries['tree_density^2']*concatted_countries['citrus_tr']
concatted_countries['tree_density^2*oil_palm_tr'] = concatted_countries['tree_density^2']*concatted_countries['oil_palm_tr']
# plt.figure
# sns.displot(concatted_countries['shade/ha'], bins=50,kde=False)
# plt.title(f'distribution plot of tree density of all countries')
# plt.show()
# concatted_countries['suitability_index'] = concatted_countries['suitability_index']/10000
# plt.figure
# fig = sm.qqplot(concatted_countries['tree_density'], line='s')
# plt.show()


# print(concatted_countries['suitability_index'].describe())
# exploratory_analysis(concatted_countries, ['suitability_index'])
concatted_countries = inter_quartiling(concatted_countries, ['suitability_index'])
concatted_countries = concatted_countries.loc[(concatted_countries['n_tree_species'] > 0)]
columns = ["Cote_d'Ivoire", 'Nigeria', 'Ghana', 'Cameroon']
# Convert dummy variables to a single 'Country' column
# concatted_countries['Country'] = concatted_countries[["Cote_d'Ivoire", 'Nigeria', 'Ghana', 'Cameroon']].idxmax(axis=1)  # Select the column where value is 1
# country = concatted_countries.groupby(by='Country')
# variables = [ 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha']
for x in ['fertilizer', '%_age 5-20']:
    print(concatted_countries[x].aggregate(['mean', 'count', 'min', 'max','median']))

# listed = [ 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
# def dummier(answer):
#     if answer == 0:
#         return answer
#     else:
#         return answer**0
# for x in listed:
#     concatted_countries.loc[:, x] = concatted_countries[x].apply(dummier)
    
# print(concatted_countries['n_tree_species'].value_counts())
# dfg = concatted_countries.groupby(by='n_tree_species')
# for x in listed:
#     print(dfg[x].value_counts())
y = concatted_countries[['yield']]

# X = concatted_countries[['tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'pesticide', 'Cameroon_low', 'Ghana_low', 'Nigeria_low', 'Cote_divoire_low', 'Ghana_medium', 'Cameroon_medium', 'Nigeria_medium', 'Cote_divoire_medium', 'Ghana_high', 'Cameroon_high', 'Cote_divoire_high', 'shade/ha']]

## without insignificant variables
# X = concatted_countries[['tree_density', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha', 'pesticide', 'Cameroon_low', 'Ghana_low', 'Nigeria_low', 'Cote_divoire_low', 'Ghana_medium', 'Cameroon_medium', 'Nigeria_medium', 'Cote_divoire_medium', 'Ghana_high', 'Cameroon_high', 'Cote_divoire_high']]
# X = concatted_countries[['tree_density', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha', 'pesticide','Ghana', 'Cameroon', "Cote_d'Ivoire"]]

## comparing cote d'ivoire with other countries
# X = concatted_countries[['tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', 'shade/ha']]
# X = concatted_countries[['tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha']]


X = concatted_countries[['tree_density', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
X = X.copy()
X['Constant'] = 1
m1 = sm.GLS(y, X).fit(cov_type='HC3')
# print(m1.summary())

# X = concatted_countries[['tree_density', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
# X = X.copy()
# X['Constant'] = 1
# m1 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m1.summary())
# # X = concatted_countries[['tree_density', 'erosion_practices', 'tree_density^2', 'weeding_level', 'fertilizer', 'suitability_index', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'pesticide','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# # X['Constant'] = 1
# # m1 = sm.OLS(y, X).fit()
# # print(m1.summary())

X = concatted_countries[['tree_density', 'tree_density^2', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
X.loc[:,'Constant'] = 1
X = X.copy()
m2 = sm.GLS(y, X).fit(cov_type='HC3')
# # print(m2.summary())

concatted_countries['tree_density_centered'] = concatted_countries['tree_density'] - concatted_countries['tree_density'].mean()
concatted_countries['tree_density_centered_sq'] = concatted_countries['tree_density_centered'] ** 2

X = concatted_countries[['tree_density_centered', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
X = X.copy()
X['Constant'] = 1
m1_2 = sm.GLS(y, X).fit(cov_type='HC3')
print(m1_2.summary())

X = concatted_countries[['tree_density_centered', 'tree_density_centered_sq', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]
X.loc[:,'Constant'] = 1
X = X.copy()
m2_2 = sm.GLS(y, X).fit(cov_type='HC3')
print(m2_2.summary())

"""Bootstrapping an idea from copilot"""
# bootstrap_results = []
# for _ in range(1000):  # 1000 bootstrap samples
#     sample = concatted_countries.sample(frac=1, replace=True)  # Resample data
#     model = sm.OLS(sample['yield'], sample[['tree_density', 'tree_density^2', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire"]]).fit()
#     bootstrap_results.append(model.params)
# print(bootstrap_results)
# # Compute confidence intervals
# lower_bound = np.percentile(bootstrap_results, 2.5, axis=0)
# upper_bound = np.percentile(bootstrap_results, 97.5, axis=0)
# print(f"Bootstrap CI: {lower_bound} to {upper_bound}")

# X = concatted_countries[['tree_density', 'tree_density^2', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire",'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X.loc[:,'Constant'] = 1
# X = X.copy()
# m2 = sm.GLS(y, X).fit(cov_type='HC3')
# print(m2.summary())

X = concatted_countries[['tree_density', 'tree_density^2', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire",'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
X.loc[:,'Constant'] = 1
X = X.copy()
m2 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m2.summary())

# # # # # # print(m2.compare_f_test(m1))

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'tree_density*n_tree_species', 'tree_density^2*n_tree_species', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', 'average_age', 'Incorporated or mulch evenly in the cocoa field', 'pesticide','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m3 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m3.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*n_tree_species', 'tree_density^2*n_tree_species', 'n_tree_species', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# print(X['suitability_index'].describe())
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m3 = sm.OLS(y, X).fit(cov_type='HC3')
# print(m3.summary())


# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', 'average_age', 'tree_density*average_age', 'Incorporated or mulch evenly in the cocoa field','pesticide','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m4 = sm.OLS(y, X).fit()
# print(m4.summary())

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'suitability_index', 'tree_density*%_age 0-5', 'tree_density^2*%_age 0-5','%_age 0-5', '%_age 20-25', '%_age >25', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m4 = sm.OLS(y, X).fit()
# print(m4.summary())

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'suitability_index', 'tree_density*%_age 20-25', 'tree_density^2*%_age 20-25','%_age 0-5', '%_age 20-25', '%_age >25', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m5 = sm.OLS(y, X).fit()
# print(m5.summary())

# X = concatted_countries[['tree_density', 'tree_density^2','fertilizer', 'tree_density*%_age >25','tree_density^2*%_age >25', '%_age 0-5', '%_age 20-25', '%_age >25', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m6 = sm.OLS(y, X).fit()
# print(m6.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*disease_level', 'tree_density^2*disease_level', 'suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m7 = sm.OLS(y, X).fit()
# print(m5.summary())

# X = concatted_countries[['tree_density', 'tree_density*pest_level', 'tree_density^2*pest_level', 'suitability_index', 'tree_density^2', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m8 = sm.OLS(y, X).fit()
# print(m7.summary())

X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*shade/ha', 'tree_density^2*shade/ha','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
X = X.copy()
X.loc[:,'Constant'] = 1
m9 = sm.OLS(y, X).fit(cov_type='HC3')
print(m9.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*n_tree_species','fertilizer', 'tree_density*average_age', 'tree_density*disease_level', 'tree_density*pest_level', 'average_age', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', 'Incorporated or mulch evenly in the cocoa field', 'pesticide','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m8 = sm.OLS(y, X).fit()

"""Regression analysis of the different trees"""
# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*medicine_tr', 'tree_density^2*medicine_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m9_2 = sm.OLS(y, X).fit()
# print(m9_2.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*N_fix_tr', 'tree_density^2*N_fix_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m10 = sm.OLS(y, X).fit()
# print(m10.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*fruit_tr', 'tree_density^2*fruit_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m11 = sm.OLS(y, X).fit()
# # print(m11.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*banana_tr', 'tree_density^2*banana_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m12 = sm.OLS(y, X).fit()
# # print(m12.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*ioly_seed_tr', 'tree_density^2*ioly_seed_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m13 = sm.OLS(y, X).fit()
# # print(m13.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*timber_tr', 'tree_density^2*timber_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m14 = sm.OLS(y, X).fit()
# # print(m14.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*citrus_tr', 'tree_density^2*citrus_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m15 = sm.OLS(y, X).fit()
# # print(m15.summary())

# X = concatted_countries[['tree_density', 'tree_density^2', 'tree_density*oil_palm_tr', 'tree_density^2*oil_palm_tr','suitability_index', 'fertilizer', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species','%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Ghana', 'Cameroon', "Cote_d'Ivoire", 'shade/ha', 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']]
# X = X.copy()
# X.loc[:,'Constant'] = 1
# m16 = sm.OLS(y, X).fit()
# print(m16.summary())

# gls_model = sm.GLS(y, X).fit()
# print(gls_model.summary())
# from io import StringIO
# df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
# df = pd.read_html(StringIO(str(soup)), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]

import scipy.stats as stats

"""Testing for normal distribution"""
## create Q-Q plot with 45-degree line added to plot
## Compute residuals
# residuals = m2.resid  

# Plot histogram
# fig, ax = plt.subplots(sharex=True, figsize=(12, 6))
# ax.hist(residuals, bins=50, edgecolor='black', density=True)
# ax.xlabel("Residuals")
# ax.ylabel("Frequency")
# ax.title("Histogram of Residuals")
# ax.grid()
# ax.show()

# # Q-Q plot
# stats.probplot(residuals, dist="norm", plot=plt)
# ax.set_xticks(np.arange(-4,4,1))
# plt.title("Q-Q Plot of Residuals")
# plt.grid(True, 'minor', color='#ddddee')
# plt.show()

# # create Q-Q plot with 45-degree line added to plot
# # Compute residuals
# residuals = m3.resid  

# Plot histogram
# plt.hist(residuals, bins=50, edgecolor='black', density=True)
# # ax.set_xticks(np.arange(-4,4,1))
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Histogram of Residuals")
# plt.grid(True, 'minor', color='#ddddee')
# plt.show()

# # Q-Q plot
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid()
# plt.show()

"""Plotting a line for tree density from slope and intercept"""
df = pd.read_html(m1.summary().tables[1].as_html(),header=0,index_col=0)[0]
summary = df.loc[:, 'coef']
df_2 = pd.read_html(m2.summary().tables[1].as_html(),header=0,index_col=0)[0]
summary_2 = df_2.loc[:, 'coef']
# [0.025   0.975]
# df.loc[df.index[2], 'P>|t|']
# summary['Constant'] + summary['suitability_index']*6864 +
#             summary['tree_density']
# x_vals = np.arange(-1.157176, 9.576750e-01, 0.04) #centered values
x_vals = np.arange(5.84, 7.97, 0.05)
variables_intercept = ('fertilizer', 'weeding_level','suitability_index', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire")
intercept_adjustment = intercept_adjustment_2 = 0
for variables in variables_intercept: 
    intercept_adjustment += df_2['coef'][variables] * concatted_countries[variables].mean() 
print(intercept_adjustment)
variables_intercept = ( 'fertilizer', 'weeding_level','suitability_index', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire")
for variables in variables_intercept: 
    intercept_adjustment_2 += df['coef'][variables] * concatted_countries[variables].mean() 


# y_vals_linear = intercept_adjustment + summary['Constant'] + summary['tree_density'] * x_vals
# y_linear_l_interval = intercept_adjustment + df['coef']['Constant']  + df['[0.025']['tree_density'] * x_vals
# y_linear_u_interval = intercept_adjustment + df['coef']['Constant']  + df['0.975]']['tree_density'] * x_vals
# 6864 * df_2['coef']['suitability_index']
y_vals_quadratic = intercept_adjustment_2 + df_2['coef']['Constant']  + df_2['coef']['tree_density'] * x_vals + df_2['coef']['tree_density^2'] * x_vals * x_vals
print(f'valua {intercept_adjustment_2 + df_2['coef']['Constant']  + df_2['coef']['tree_density'] * np.log(996) + df_2['coef']['tree_density^2'] * np.log(996) * np.log(996)}')
# y_quadratic_l_interval = intercept_adjustment_2 + df_2['coef']['Constant']  + df_2['coef']['tree_density']  * x_vals + df_2['[0.025']['tree_density_centered_sq'] * x_vals * x_vals
# y_quadratic_u_interval = intercept_adjustment_2 + df_2['coef']['Constant']  + df_2['coef']['tree_density'] * x_vals + df_2['0.975]']['tree_density_centered_sq'] * x_vals * x_vals



x_vals_un_logged = np.exp(x_vals)  # Reverse log transformation
# Find the index of the maximum value in y
max_index = np.argmax(y_vals_quadratic)

# # Get the x and y values at the maximum index
max_x = x_vals_un_logged[max_index]
max_y = y_vals_quadratic[max_index]
# plt.scatter(concatted_countries['tree_density'], y, label="Data Points")
# plt.plot(x_vals, y_vals_linear, '-', color='c', label='Linear')
# plt.plot(x_vals, y_linear_l_interval, '.', color='c', label='Lower bound', linewidth = 1)
# plt.plot(x_vals, y_linear_u_interval, '.', color='c', label='Upper bound', linewidth = 1)

plt.annotate(f'{int(max_x)} Cacao trees/ha \n {int(max_y)} kg/ha of cacao', xy=(max_x, max_y), xytext=(max_x + 200, max_y - 45),
         arrowprops=dict(facecolor='r'))

plt.plot(x_vals_un_logged, y_vals_quadratic, '-', color='navy', label='Quadratic')
# plt.plot(x_vals, y_quadratic_l_interval, '.', color='b',label='Lower bound', linewidth = 1)
# plt.plot(x_vals, y_quadratic_u_interval, '.', color='b',label='Upper bound', linewidth = 1)
# plt.scatter(X['tree_density'], y, label="Data Points")
plt.xlabel("Cacao tree density (trees/ha)")
plt.ylabel("Yield (kg/ha)")
plt.title(f"The cacao tree density for the optimum yield level", fontsize=16)
# plt.legend()
plt.show

"""Printing summary model"""
# model = summary_col([m2, m9_2, m10, m11, m12, m13, m14, m15, m16],stars=True,float_format='%0.3f',
#                   model_names=['model 4','model 5', 'model 6', 'model 7', 'model 8', 'model 9', 'model 10', 'model 11', 'model 12'],
#                   info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
#                             'R2':lambda x: "{:.3f}".format(x.rsquared)})
# print(model.as_latex())

"""Correlation matrix"""
# # # multi_c_df = concatted_countries[['tree_density', 'erosion_practices', 'weeding_level', 'prune_level', 'disease_level', 'pest_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field','Cameroon_low', 'Ghana_low', 'Nigeria_low', 'Cote_divoire_low', 'Ghana_medium', 'Nigeria_medium', 'Cameroon_medium', 'Cote_divoire_medium', 'Ghana_high', 'Cameroon_high', 'Cote_divoire_high', 'shade/ha']]
# multi_c_df = concatted_countries[['tree_density', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', 'n_tree_species', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'shade/ha','Ghana', 'Cameroon', "Cote_d'Ivoire", 'medicine_tr', 'N_fix_tr', 'banana_tr',  'oil_palm_tr']]

# correlation_matrix = multi_c_df.corr()
# plt.figure(figsize=(10, 6))
# # Create a heatmap for the correlation matrix
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# # Title for the heatmap
# plt.title("Correlation Heatmap", fontsize=16)
# # Show the heatmap
# plt.show()
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# """calculating vif data"""
# vif_data = pd.DataFrame()
# vif_data["feature"] = multi_c_df.columns
# # Calculate VIF and round to 4 decimal places
# vif_data["VIF"] = [round(variance_inflation_factor(multi_c_df.values, i), 4) for i in range(multi_c_df.shape[1])]
# # Sort VIF values in descending order
# vif_data = vif_data.sort_values(by="VIF", ascending=False)
# # Display the VIF DataFrame
# print(vif_data)

# # # Init plots
# # fig, ax = plt.subplots(figsize=(8,6))
# print(concatted_countries['tree_density_centered'].describe())
"""Making plots from the regression table"""
def regression_plots(name_variable, df, coef_interaction, coef_quad_interaction, levels):
    # Define the range of tree density values
    tree_density_range = np.linspace(5.84, 7.97, 100)  # 100 values for smooth plotting
    variables_intercept = ('n_tree_species', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire")
    intercept_adjustment = 0
    for variables in variables_intercept: 
        intercept_adjustment += df['coef'][variables] * concatted_countries[variables].mean() 
        print(variables)
        print(df['coef'][variables] * concatted_countries[variables].mean())
    # intercept_adjustment += df['coef']['tree_density^2'] * concatted_countries['tree_density'].mean()* concatted_countries['tree_density'].mean() + df['coef'][coef_interaction]* concatted_countries['tree_density'].mean()* concatted_countries[name_variable].mean() + df['coef'][coef_quad_interaction]* concatted_countries['tree_density'].mean()* concatted_countries['tree_density'].mean()* concatted_countries[name_variable].mean()

    # Convert log(tree_density) back to original tree_density scale
    tree_density = np.exp(tree_density_range)  # Reverse log transformation
    # Plot regression lines for each disease level
    plt.figure(figsize=(8, 6))

    for level in levels:
        # Compute yield based on regression equation
        predicted_yield = (
            intercept_adjustment + df['coef']['Constant'] +
            df['coef']['tree_density'] * tree_density_range +
            df['coef'][name_variable] * level +
            df['coef'][coef_interaction] * tree_density_range * level + df['coef'][coef_quad_interaction] * tree_density_range * tree_density_range * level +
            df['coef']['tree_density^2'] * tree_density_range * tree_density_range
        )
        
        # Plot each disease level line
        plt.plot(tree_density, predicted_yield, label=f"{level} trees/ha")
        max_index = np.argmax(predicted_yield)
        max_x = tree_density[max_index]
        max_y = predicted_yield[max_index] 
        plt.annotate(int(max_x), xy=(max_x, max_y), xytext=(max_x -300, max_y - 25), arrowprops=dict(facecolor='b'))
    # Formatting
    plt.xlabel("Cacao tree density (trees/ha)")
    plt.ylabel("Yield (kg/ha)")
    plt.title(f"The interaction effect of cacao tree density and shade tree density with Yield")
    plt.legend(title='Number of shade trees')
    plt.grid(True)
    plt.show()

"""make regression plot for the different variables"""
def results_summary_to_dataframe(results):
        '''take the result of an statsmodel results table and transforms it into a dataframe'''
        pvals = results.pvalues
        coeff = results.params

        results_df = pd.DataFrame({"P>|t|":pvals,
                                "coef":coeff,
                                    })

        #Reordering...
        results_df = results_df[["coef","P>|t|"]]
        return results_df
df = results_summary_to_dataframe(m9)
# # print(df)
# # # regression_plots('Pest level', 254.3781, 0.0749 , 15.2326, -0.0112, -2.028e-05, [0, 1, 2, 3])
# # # regression_plots(name_variable, constant, coef_tree_density, coef_variable, coef_interaction, coef_tree_density_2, levels)
# regression_plots('shade/ha', df, 'tree_density*shade/ha', 'tree_density^2*shade/ha', [15, 10, 5, 0])
# regression_plots("Age 0-5", -1358.468 + 6864 * 0.38, 389.557, 7.183, -0.927, -26.969, [0, 25, 50, 75, 100])
# regression_plots('n_tree_species', df, 'tree_density*n_tree_species', 'tree_density^2*n_tree_species', [1, 2, 3, 4])
# regression_plots('Number of tree species', -1104.9384 + 255, 387.6407, -193.9219, 26.4903, -31.6245 , [1, 2, 3, 4])
# regression_plots("disease level", constant, coef_tree_density, coef_variable, coef_interaction, coef_tree_density_2, [0, 1, 2, 3])
# regression_plots("age", constant, coef_tree_density, coef_variable, coef_interaction, coef_tree_density_2, [5, 10, 20, 30])

"""Making plots from the regression table for tree_species"""
def plot_tree_diversity():
    # Define the range of tree density values
    def results_summary_to_dataframe(results):
        '''take the result of an statsmodel results table and transforms it into a dataframe'''
        pvals = results.pvalues
        coeff = results.params

        results_df = pd.DataFrame({"P>|t|":pvals,
                                "coeff":coeff,
                                    })

        #Reordering...
        results_df = results_df[["coeff","P>|t|"]]
        return results_df
    tree_density_range = np.linspace(5.84, 7.97, 100)  # 100 values for smooth plotting
    variables_intercept = ['n_tree_species', 'fertilizer', 'suitability_index', 'weeding_level', 'prune_level', 'disease_level', '%_age 0-5', '%_age 20-25', '%_age >25', 'Incorporated or mulch evenly in the cocoa field', 'Ghana', 'Cameroon', "Cote_d'Ivoire", 'medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
    
    
    # Convert log(tree_density) back to original tree_density scale
    tree_density = np.exp(tree_density_range)  # Reverse log transformation
    # Plot regression lines for each disease level
    plt.figure(figsize=(8, 6))
    regressions = [m9_2, m10, m11, m12, m13, m14, m15, m16]
    name_variable = ['medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
    legend_name = ['Medicinal', 'Nitrogen fixing', 'Fruit', 'Banana', 'Oily seed', 'Timber', 'Citrus', 'Palm oil']
    for counter in range (len(regressions)):
        # line_score = pd.read_html(StringIO(str(soup)), attrs = {'id': 'line_score'})[0]
        # df = pd.read_html(str(soup), attrs={"id": f"box-{team}-game-{stat}"}, index_col=0)[0]
        intercept_adjustment = 0
        
        df = results_summary_to_dataframe(regressions[counter])
        
        # print(df)
        # print(df[0])
        summary = df.loc[:, 'coeff']
        # Compute yield based on regression equation
        # tree_density                                     1213.061154   7.383812e-21
        # tree_density^2                                    -86.001837   1.521953e-20
        # tree_density*oil_palm_tr                         -603.181873   3.847365e-02
        # tree_density^2*oil_palm_tr                         43.509966   3.724898e-02
        for variables in variables_intercept: 
            intercept_adjustment += summary[variables] * concatted_countries[variables].mean() 
        intercept_adjustment_tree = intercept_adjustment - summary[name_variable[counter]] * concatted_countries[name_variable[counter]].mean()
        predicted_yield = (
            summary['Constant'] + intercept_adjustment_tree +
            summary['tree_density'] * tree_density_range +
            summary.iloc[counter + 18] +
            summary.iloc[2] * tree_density_range +  summary.iloc[3] * tree_density_range * tree_density_range + 
            summary['tree_density^2'] * tree_density_range * tree_density_range
        )
        # print(df.iloc[counter + 9][0])
        # Plot each disease level line
        if (df.loc[df.index[2], 'P>|t|'] < 0.05):
        #     print(df.loc[df.index[counter + 18]])
            plt.plot(tree_density, predicted_yield, '-', label=f"{legend_name[counter]}")
        #     max_index = np.argmax(predicted_yield)
        #     max_x = tree_density[max_index]
        #     max_y = predicted_yield[max_index] 
            # plt.annotate(int(max_x), xy=(max_x, max_y), xytext=(max_x, max_y - 10))
        # else:
            # plt.plot(tree_density, predicted_yield, '-', label=f"{legend_name[counter]}")
    
    # Formatting
    plt.xlabel("Cacao tree Density (trees/ha)")
    plt.ylabel("Yield (kg/ha)")
    plt.title(f"The interaction effects of cacao tree density and different shade trees on yield")
    # reordering the labels
    # handles, labels = plt.gca().get_legend_handles_labels()

    # # specify order
    # order = [2, 3, 1, 0]

    # # pass handle & labels lists along with order as below
    # plt.legend([handles[i] for i in order], [labels[i] for i in order])

    plt.legend(title='Tree type')
    plt.grid(True)
    plt.show()

"""make regression plot for the different tree_species"""
# df = pd.read_html(m9_2.summary().tables[1].as_html(),header=0,index_col=0)[0]
# # print(df[0])
# summary = df.loc[:, 'coef']
# print(summary['tree_density'])
# print(summary[4])
# print(summary)
# plot_tree_diversity()

# print(np.exp(5.846823))
# print(np.exp(7.961675))
# print(np.exp(7.004000))
# print(np.exp(7.013116))

"""Plotting a line for shade tree_species from slope and intercept"""
# df = results_summary_to_dataframe(m2)
# # summary_2 = df_2.loc[:, 'coef']

# # summary['Constant'] + summary['suitability_index']*6864 +
# #             summary['tree_density']
# x_vals = np.arange(0, 1.1, 0.1)
# # x_vals_un_logged = np.exp(x_vals) 
# name_variable = ['medicine_tr', 'N_fix_tr', 'fruit_tr', 'banana_tr', 'ioly_seed_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr']
# legend_name = ['Medicinal', 'Nitrogen fixing', 'Fruit', 'Banana', 'Oily seed', 'Timber', 'Citrus', 'Palm oil']
# for counter in range (len(name_variable)):
#     y_vals_linear = (df['coef']['Constant'] + 6864 * df['coef']['suitability_index'] + 7 * df['coef']['tree_density'] + 
#                     7 * 7 * df['coef']['tree_density^2'] + # suitability index and treedensity is included to make the plot start at a logical point in the graph
#                     df['coef'][name_variable[counter]] * x_vals)
#     if (df['P>|t|'][name_variable[counter]] < 0.05):
#         plt.plot(x_vals, y_vals_linear, '-', label=legend_name[counter])
#         plt.xlim(xmin=0, xmax=1)
#     else:
#         plt.plot(x_vals, y_vals_linear, '.', label=legend_name[counter])
#         plt.xlim(xmin=0, xmax=1)
# # plt.scatter(X['tree_density'], y, label="Data Points")
# plt.xlabel("From 0 tree to 1 type of shade tree")
# plt.ylabel("Yield (kg/ha)")
# plt.title(f"The linear relationship of the shade tree species with yield", fontsize=16)
# ## reordering the labels
# handles, labels = plt.gca().get_legend_handles_labels()

# # specify order
# order = [1, 4, 2, 5, 6, 7, 0, 3]

# # pass handle & labels lists along with order as below
# plt.legend([handles[i] for i in order], [labels[i] for i in order], title='Tree type')
# plt.show


# df = results_summary_to_dataframe(m2)
# name_variable = ['N_fix_tr','ioly_seed_tr', 'fruit_tr', 'timber_tr', 'citrus_tr', 'oil_palm_tr', 'medicine_tr', 'banana_tr']

# legend_name = ['Nitrogen fixing', 'Oily seed',  'Fruit',  'Timber', 'Citrus', 'Palm oil', 'Medicinal', 'Banana']
# for counter in range (len(name_variable)):
#     y_vals_linear = (df['coef'][name_variable[counter]])
#     if (df['P>|t|'][name_variable[counter]] < 0.05):
#         plt.bar(legend_name[counter], y_vals_linear, color='b', label="Significant")
#         # plt.xlim(xmin=0, xmax=1)
#     else:
#         plt.bar(legend_name[counter], y_vals_linear, color='0.8', label="Insignificant")
#         # plt.xlim(xmin=0, xmax=1)
#     if (df['coef'][name_variable[counter]] < 0.0):
#         plt.text(legend_name[counter], y_vals_linear - 1.5, int(y_vals_linear), ha='center')
#     else:
#         plt.text(legend_name[counter], y_vals_linear + 0.5, int(y_vals_linear), ha='center')
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.legend(handles[3:5], labels[3:5])
# # plt.scatter(X['tree_density'], y, label="Data Points")
# plt.xticks(rotation=45, ha="right")
# plt.xlabel("Type of tree")
# plt.ylabel("Yield difference (kg/ha)")
# plt.title(f"The change in yield for different shade trees compared to a full sun system", fontsize=12)
# # pass handle & labels lists along with order as below
# plt.show


