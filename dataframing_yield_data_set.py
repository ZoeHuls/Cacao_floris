## setting up file
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
#df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_questions_cocoa_compressed.csv')
df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_questions_yield_compressed.csv')

ques = df.groupby(by='question')
q = ques.groups.keys()

questions_answered = [ 'COCOA PRODUCTION IN DRY\nHow many kg of dry cocoa beans did you produce on all cocoa plots during the last full crop cycle?', 'How many cocoa plots do you have? (include rented plots and sharecropping)', 'How many of these plots are greater than 4 ha?', 'How many people in your household share resources and thus depend on your income for their livelihood? (include yourself, other adults, children, and elderly)','ONLY FOR BRAZIL, PNG, CDI, ECU AND UGA\nCOCOA PRODUCTION IN WET\nHow many kg of wet cocoa beans did you produce on all cocoa plots during the last full crop cycle?', 'What is the total estimated area (ha) of all your cocoa plots ?']

# attempt to create variable of each question, worked, only question and answer remained and thus remained duplicated
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']
variable_names = ['dry_produced', 'many_plots', 'n_plots_>4ha','n_hh_members_depend_income','wet_produced','farm_ha']


merging_dataframe = pd.DataFrame(df['olam_farmer_id'])

for count in range(6):
    df_question =  ques.get_group(questions_answered[count])
    prepared1 = df_question[['olam_farmer_id','answer']]
    prepared1 = prepared1.rename(columns={'answer':variable_names[count]})
    merging_dataframe = merging_dataframe.merge(prepared1, on = 'olam_farmer_id', how = 'outer')

## dropping duplicates from both dataframes to make merge go faster
data_drop = merging_dataframe.drop_duplicates(subset=['olam_farmer_id'])
df1 = df.drop(columns = ['question','answer'])
df1_drop = df1.drop_duplicates(subset=['olam_farmer_id'])

## Merging the created dataframe with existing dataset with farmer info now still prints duplicates
final = df1.merge(data_drop, on = 'olam_farmer_id', how = 'inner')
final_drop = df1_drop.merge(data_drop, on = 'olam_farmer_id', how = 'inner')
final_drop[variable_names] = final_drop[variable_names].apply(pd.to_numeric, errors='coerce')
print(final_drop.head())
final_drop['dry_produced' => 800000] 

# Create the box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='country_name', y='dry_produced', data=final_drop)
plt.title('Distribution of Dry Produced by Country')
plt.xticks(rotation=45)
plt.show()