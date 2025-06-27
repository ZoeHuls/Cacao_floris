import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# df = pd.read_csv("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_concat_2025.csv")
df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')
#df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\all_farmer_survey_compressed.csv')

questions = ['What percentage of your cocoa trees are aged 0 to 5 years?', 'What percentage of your cocoa trees are aged 5-20 years?', 'What percentage of your cocoa trees are aged 20-25 years?', 'What percentage of your cocoa trees are over 25 years?']
variable_names = ['%_age 0-5','%_age 5-20','%_age 20-25','%_age >25']
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name']
all_heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']
# merging_dataframe = pd.DataFrame(index=all_heading)

red_question = ['What is the total amount of fungicides applied to all your cocoa plots in the last crop cycle? (report in grams if non-liquid, and non-diluted liters if liquid)', 'What is the total amount of insecticide applied to all your cocoa plots in the last crop cycle?  (report in grams if non-liquid, and non-diluted liters if liquid)','What is the total amount of chemical herbicide applied to all your cocoa plots in the last crop cycle? If liquid, please answer the number of non-diluted liters of herbicide used.']
#df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\database_yield_compressed.csv')

ques = df.groupby(by="question")
q = ques.groups.keys()
for x in red_question:
    print(x)
    df_question = ques.get_group(x)
    print(df_question.head())
    answer = df_question.groupby(by="answer")
    q = answer.groups.keys()
    print(q)
    x = list(q)
    listed = [i.split(' || ') for i in x]

    # # list = q.split(" || ")
    def flatten_comprehension(matrix):
        return [item for row in matrix for item in row]

    flattened = flatten_comprehension(listed)
    mylist = list(dict.fromkeys(flattened))
    print(mylist) 

    # print(df_question.head())
    # print(df_question["answer"].describe())
    print(df_question["answer"].value_counts(dropna = False))
    # df_question['answer'] = pd.to_numeric(df_question['answer'], errors='coerce')
    # print(df_question['answer'].describe())
    # def inter_quartiling(dfg, column):
    #     country = dfg.iloc[1, 3]
    #     data = dfg #.loc[~(dfg[column] == 0.000000e+00)]
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

    # df_question = df_question.groupby(by= 'country_name', group_keys=False).apply(lambda dfg: inter_quartiling(dfg, 'answer'))
    # #df_question = df_question.loc[(df_question['answer'] > 25) & (df_question['answer'] < 100)]

    # print(df_question['answer'].aggregate(['mean', 'count', 'min', 'max','median']))
    # # sns.displot(df_question['answer'], bins=100,kde=False)
    # # plt.title(f'distribution plot of oldest tree ages')

    # # merging_dataframe = df.iloc[0:1]
    # # merging_dataframe = merging_dataframe.drop(columns = ['question','answer'])
    # # print(merging_dataframe)

    # # for count in range(4):
    # #     df_question = ques.get_group(questions[count])
    # #     df_question['answer'] = df_question['answer'].apply(pd.to_numeric, errors='coerce')
    # #     df_question = df_question.rename(columns={'answer': variable_names[count]})
    # #     df_question = df_question.drop(columns = ['question'])
    # #     dataframe = merging_dataframe.merge(df_question, how = 'outer', left_on= heading, right_on= heading)
    # #     merging_dataframe = dataframe

    # # print(dataframe.tail())
    # # dataframe.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\percentage_age_compressed.csv', index=False)




