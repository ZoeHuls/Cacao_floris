## setting up file
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
#df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_questions_cocoa_compressed.csv')
df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_questions_yield_compressed.csv')

ques = df.groupby(by='question')
q = ques.groups.keys()
#print(q)

#yield_ha['answer'].astype(str).astype(int)
#yield_ha['answer'] = yield_ha['answer'].convert_dtypes()
#yield_ha['answer'] = pd.to_numeric(yield_ha['answer'], errors='coerce')
#yield_ha['answer'] = pd.to_numeric(yield_ha['answer'], errors='coerce').convert_dtypes() 

## Prints a summary for each question
# def summary(group, ques):
#     df = ques.get_group(group)
#     yield_ha1 = df.convert_dtypes()
#     yield_ha1['answer'] = pd.to_numeric(yield_ha1['answer'], errors='coerce')
#     print(group)
#     print(yield_ha1['answer'].aggregate(['mean', 'count', 'min', 'max']))

questions_answered = [ 'COCOA PRODUCTION IN DRY\nHow many kg of dry cocoa beans did you produce on all cocoa plots during the last full crop cycle?', 'How many cocoa plots do you have? (include rented plots and sharecropping)', 'How many of these plots are greater than 4 ha?', 'How many people in your household share resources and thus depend on your income for their livelihood? (include yourself, other adults, children, and elderly)','ONLY FOR BRAZIL, PNG, CDI, ECU AND UGA\nCOCOA PRODUCTION IN WET\nHow many kg of wet cocoa beans did you produce on all cocoa plots during the last full crop cycle?', 'What is the total estimated area (ha) of all your cocoa plots ?']
print(df.iloc[0, 8])
def preparing(group, ques):
    dataframe = ques.get_group(group)
    #yield_ha1 = df.loc[:, df.columns != 'farmer_id'].convert_dtypes()
    dataframe.loc['answer'] = pd.to_numeric(dataframe['answer'], errors='coerce', downcast='integer')
    return dataframe 
# heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']

# df['answer'].convert_dtypes()
# print(type(df.head()))
# print(type(df.iloc[3, 8]))
# print("pause")
# preparing('COCOA PRODUCTION IN DRY\nHow many kg of dry cocoa beans did you produce on all cocoa plots during the last full crop cycle?', ques)
# df.loc['answer'] = pd.to_numeric(df['answer'], errors='coerce',downcast='integer')
# print(type(df.iloc[3, 8]))

## attempt if I can create a new variable dry_produced for the dataframe 
# dry_produced = preparing('COCOA PRODUCTION IN DRY\nHow many kg of dry cocoa beans did you produce on all cocoa plots during the last full crop cycle?',ques)
# dry_produced1 = dry_produced.drop(columns = ['question'])
# dry_produced1 = dry_produced1.rename(columns={"answer":"dry_produced"})
# df = df.merge(dry_produced1, on = heading, how = 'left')
# print(df.head())

# attempt to create variable of each question, worked, only question and answer remained and thus remained duplicated
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']
variable_names = ['dry_produced', 'many_plots', 'n_plots_>4ha','n_hh_members_depend_income','wet_produced','farm_ha']
df1 = df.drop(columns = ['question','answer'])

merging_dataframe = pd.DataFrame(df['olam_farmer_id'])

# for count in range(6):
#     question = questions_answered[count]
#     prepared = preparing(question, ques)
#     prepared1 = prepared[['olam_farmer_id','answer']]
#     prepared1 = prepared1.rename(columns={'answer':variable_names[count]})
#     merging_dataframe = merging_dataframe.merge(prepared1, on = 'olam_farmer_id', how = 'outer')

for count in range(6):
    df_question =  ques.get_group(questions_answered[count])
    prepared1 = df_question[['olam_farmer_id','answer']]
    prepared1 = prepared1.rename(columns={'answer':variable_names[count]})
    merging_dataframe = merging_dataframe.merge(prepared1, on = 'olam_farmer_id', how = 'outer')


data_drop = merging_dataframe.drop_duplicates(subset=['olam_farmer_id'])
df1_drop = df1.drop_duplicates(subset=['olam_farmer_id'])
## Merging the created dataframe with existing dataset with farmer info now still prints duplicates
final = df1.merge(data_drop, on = 'olam_farmer_id', how = 'inner')
final_drop = df1_drop.merge(data_drop, on = 'olam_farmer_id', how = 'inner')

last_6 = final_drop.iloc[7:13]
final_drop[variable_names] = final_drop[variable_names].apply(pd.to_numeric, errors='coerce')
#final_drop['farm_ha'] = pd.to_numeric(final_drop['farm_ha'], errors='coerce')
print(final_drop.head())
#print(final_drop.tail())
print(type(final_drop.iloc[2, 11]))
print(final_drop.iloc[2, 11])
print(final_drop.dtypes)
# for x in questions_answered:
#     summary(x, ques)

# for x in questions_answered:
#     prepared = preparing(x, ques)

# print(prepared.head())
#df['answer'].convert_dtypes()
#yield_ha1 = df.loc[:, df.columns != 'farmer_id'].convert_dtypes() 

## trying to pivot the table TypeError: agg function failed [how->mean,dtype->object]
# df = ques.get_group('COCOA PRODUCTION IN DRY\nHow many kg of dry cocoa beans did you produce on all cocoa plots during the last full crop cycle?')
# df.loc['answer'] = pd.to_numeric(df['answer'], errors='coerce')
# df_pivot = df.pivot_table(index = 'farmer_id' ,
#                     columns = 'question',
#                     values = 'answer')

# print(df_pivot.head())