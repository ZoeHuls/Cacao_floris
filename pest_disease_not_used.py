from functools import reduce
pd.set_option('display.max_colwidth', None)
countries = ["COTE D'IVOIRE",'GHANA', "NIGERIA","CAMEROON"]
important_columns = ['farm_ha','dry_produced']
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name']
all_pests = ['Aphids', 'Beetles', 'Bugs', 'Caterpillar', 'Cocoa Pod Borer', 'Cocoa mosquito', 'Cocoa stem borers', 'Gem Mite', 'Grasshoppers', 'Green moss', 'Mealybugs', 'Mirids', 'Moths', 'None', 'Other', 'Parasitic plant/ mistletoes/ epiphytes', 'Psyllids', 'Rats', 'Skeletonisers', 'Stink or shield bugs', 'Termites', 'Thrips']
different_pests = ['Aphids', 'Beetles', 'Bugs', 'Caterpillar', 'Cocoa Pod Borer', 'Cocoa mosquito', 'Cocoa stem borers', 'Gem Mite', 'Grasshoppers', 'Green moss', 'Mealybugs', 'Mirids', 'Moths', 'None', 'Other', 'Parasitic plant/ mistletoes/ epiphytes', 'Psyllids', 'Rats', 'Skeletonisers', 'Stink or shield bugs', 'Termites', 'Thrips']
different_diseases = ['Anthracnose', 'Black pod disease', 'CaMMV', 'Cancer of Lasiodiplodia', 'Charcoal pod rot', 'Frosty Pod Rot', 'None', 'Other', 'Pink Disease', 'Stem canker', 'Swollen Shoot Virus', 'Thread blights of cocoa', 'Warty pod'] 
ques = df.groupby(by='question')

# df_pest = ques.get_group('What was the most damaging pest in your cocoa plots in the last crop cycle?')
# print(df_pest.info())
# df_pest.dropna(subset=['answer'], inplace= True)
# for x in different_pests:
#     df_pest[x] = df_pest['answer'].str.contains(x)
#     #df_pest = df_pest.assign(x = df_pest['answer'].str.contains(x))
#     df_pest[x] = df_pest[x].astype(int)
#     Pest_clean = df_pest.drop(columns = ['question', 'answer'])
#     print(Pest_clean[x].value_counts())

# df_disease = ques.get_group('What was the most damaging disease in your cocoa plots in the last crop cycle?')
# print(df_disease.info())
# df_disease.dropna(subset=['answer'], inplace= True)
# for x in different_diseases:
#     df_disease[x] = df_disease['answer'].str.contains(x)
#     #df_pest = df_pest.assign(x = df_pest['answer'].str.contains(x))
#     df_disease[x] = df_disease[x].astype(int)
#     # df_disease = df_disease.assign(x = df_disease['answer'].str.contains(x))
#     # df_disease[x] = df_disease[x].astype(int)
#     Disease_clean = df_disease.drop(columns = ['question', 'answer'])

def inter_quartiling(dfg, column):
    data = dfg.loc[~(dfg[column] == 0.000000e+00)]
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75) 
    IQR = Q3 - Q1
  
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_outlied = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]   
    return df_outlied

# ## removing outliers from production and farm_ha and printing distribution 
for column in important_columns:  
    df_yield = df_yield.groupby(by='country_name', group_keys=False).apply(lambda dfg: inter_quartiling(dfg, column))
df_yield['yield'] = df_yield['dry_produced']/df_yield['farm_ha']
df_yield = df_yield.groupby(by= 'country_name', group_keys=False).apply(lambda dfg: inter_quartiling(dfg, 'yield'))
# data_frames = [Pest_clean, df_yield]
# data = reduce(lambda  left,right: pd.merge(left,right, on= heading, how='outer'), data_frames)
# data = Pest_clean.merge(Disease_clean, how = 'outer', left_on= heading, right_on= heading)
# # data.isnull().sum().sort_values(ascending = False)
# data_d = data.drop(columns = ['Bugs', 'Cocoa stem borers', 'Gem Mite', 'Grasshoppers', 'Moths', 'None_x', 'Other_x','None_y', 'Other_y', 'Psyllids', 'Skeletonisers', 'Thrips', 'CaMMV', 'Cancer of Lasiodiplodia', 'Stem canker', 'Warty pod'])
# datanan = data_d.dropna(axis=0, how='all')
# datanan_drop = datanan.drop_duplicates(subset=['olam_farmer_id'])

## function printing average yield from different categories, value counts per country, general yield plot per country without NaN values of particular variable
def exploratory_analysis (df, columns, countries):
    for column in columns:    
        plt.figure()
        sns.boxplot(x=column, y='yield', data=df)
        plt.title(f'Box Plot without Outliers of {column}')
        plt.show()
        
        country = df.groupby(by='country_name', group_keys=False)
        country[column].value_counts().plot(kind= 'barh')
        for cntr in countries:
            df_country = country.get_group(cntr)
            plt.figure()
            sns.boxplot(x=column, y='yield', data=df_country)
            plt.title(f'Box Plot without Outliers of {column} of {cntr}')
            plt.show()
            print(df_country[column].value_counts(dropna = False))
    yield_measure = df.dropna(subset=columns)
    plt.figure()
    sns.boxplot(x='country_name', y='yield', data=yield_measure)
    plt.title(f'Box Plot without Outliers of yield NaN of {columns} excluded')
    plt.show()

# exploratory_analysis(datanan, different_pests, countries)

# datanan_drop.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\Pest&disease.csv', index=False)
df_pest_disease = ques.get_group('What pests and diseases were present in your cocoa plots in the last crop cycle?')
pest_disease = ['Ant', 'Frosty Pod Rot', 'Grasshoppers', 'Thrips', 'Anthracnose', 'Black pod disease', 'Caterpillar', 'Cocoa Pod Borer', 'Cocoa stem borers', 'Green moss', 'Mealybugs', 'Mirids', 'Parasitic plant/ mistletoes/ epiphytes', 'Stem canker', 'Stink or shield bugs', 'Termites', 'None', 'Other', 'Aphids', 'Beetles', 'Cocoa mosquito', 'Moths', 'Charcoal pod rot', 'Thread blights of cocoa', 'Pink Disease', 'Skeletonisers', 'Warty pod', 'Bugs', 'Swollen Shoot Virus', 'Rats', 'Psyllids', 'Machete disease (Ceratocistis)', 'Root disease (Roselina pepo)']
df_pest_disease.dropna(subset=['answer'], inplace= True)
for x in pest_disease:
    df_pest_disease[x] = df_pest_disease['answer'].str.contains(x)
    #df_pest = df_pest.assign(x = df_pest['answer'].str.contains(x))
    df_pest_disease[x] = df_pest_disease[x].astype(int)
    Pest_clean = df_pest_disease.drop(columns = ['question', 'answer'])
    print(Pest_clean[x].value_counts())
data = Pest_clean.merge(df_yield, how = 'outer', left_on= heading, right_on= heading)
datanan = data.dropna(axis=0, how='all')
exploratory_analysis(datanan, pest_disease, countries)

## making random forest disease
# training_data, testing_data = train_test_split(datanan, test_size=0.3, random_state=25)
# x_train = training_data.drop(columns = ['olam_farmer_id','farmer_id', 'Gender', 'country_name', 'region_name', 'district_name', 'place_name', 'dry_produced', 'many_plots', 'n_plots_>4ha', 'n_hh_members_depend_income', 'wet_produced', 'farm_ha', 'yield'])
# y_train = training_data[['yield']]
# y_train = y_train.to_numpy() #here we convert y_train from a pandas dataframe with one column to a numpy array.

# x_test = testing_data.drop(columns = ['olam_farmer_id','farmer_id', 'Gender', 'country_name', 'region_name', 'district_name', 'place_name', 'dry_produced', 'many_plots', 'n_plots_>4ha', 'n_hh_members_depend_income', 'wet_produced', 'farm_ha', 'yield'])
# y_test = testing_data[['yield']]
# y_test = y_test.to_numpy().ravel()

# rf = RandomForestRegressor(n_estimators = 75, max_features = 'sqrt', max_depth = 3, random_state = 18).fit(x_train, y_train)
# # firsttree = rf.estimators_[0]
# # plt.figure(figsize=(22,8))
# # plot_tree(firsttree, feature_names=x_train.columns, filled=True, fontsize = 11)
# # plt.title('title')
# # plt.show()
# y_pred = rf.predict(x_test)

# fst = 12
# plt.plot(y_test,'bo')
# plt.plot(y_pred,'ro')
# plt.legend(['Observed','Predicted'], fontsize = fst)
# plt.xticks(fontsize=fst)
# plt.yticks(fontsize=fst)
# plt.ylabel('yield',fontsize=fst)

# list_mse = []
# list_r2 = []

# for n_est in [75, 150]:
#     for dep in [4, 7, 10]:
#        rf = RandomForestRegressor(n_estimators = n_est, max_features = 'sqrt', max_depth = dep, random_state = 18).fit(x_train, y_train)
#        y_pred = rf.predict(x_test)
#        mse = mean_squared_error(y_test, y_pred)
#        r2 = r2_score(y_test, y_pred)
#        list_mse.append(mse)
#        list_r2.append(r2)

# print(list_mse)
# print(list_r2)

# rf = RandomForestRegressor(n_estimators= 200, max_features = 'sqrt', max_depth = 12, random_state = 18).fit(x_train, y_train) #fill in hyperparameters

# y_pred = rf.predict(x_test)
# fst = 12
# plt.plot(y_test,'bo')
# plt.plot(y_pred,'ro')
# plt.legend(['Observed','Predicted'], fontsize = fst)
# plt.xticks(fontsize=fst)
# plt.yticks(fontsize=fst)
# plt.ylabel('yield',fontsize=fst)

# rf = RandomForestRegressor(n_estimators = 200, max_features = 'sqrt', max_depth = 12, random_state = 18).fit(x_train, y_train)
# sorted_idx = rf.feature_importances_.argsort() #we sort the importance of features from high to low.
# plt.figure(figsize=(7, 10.5))
# plt.barh(x_train.columns[sorted_idx], rf.feature_importances_[sorted_idx])
# plt.show()
