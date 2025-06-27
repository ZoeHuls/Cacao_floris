import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\countries_practice_compressed.csv')
import matplotlib.pyplot as plt
from functools import reduce
pd.set_option('display.max_colwidth', None)
heading = ['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name']
ques = df.groupby(by='question')

## variable for the surrounding
surroundings = ['Beehives or insect hotels', 'Living fences or hedgerows', 'buffers', 'Native vegetation and flowers strips', 'None of the above', 'Permanently protected forest areas', 'Ponds or wetlands']
surrounding = ques.get_group('Do you have any of the following natural areas in your cocoa plots or their immediate surroundings?')
surrounding_dropped = surrounding.dropna(subset=['answer'])
for x in surroundings:
    # df.loc[df['average_age'] < 5, 'young'] = 1
    #df_management[x] = df_management['answer'].str.contains(x)
    surrounding_dropped[x] = surrounding_dropped['answer'].str.contains(x)
    #management.assign(xmanagement['answer'].str.contains(x)
    surrounding_dropped[x] = surrounding_dropped[x].astype(int)
surrounding_clean = surrounding_dropped.drop(columns = ['question', 'answer'])
for x in surroundings:
    print(surrounding_dropped[x].value_counts(dropna = False))
print(surrounding_clean.head())


# variable for education
education = ques.get_group('Please ask the farmer to read aloud the following passage: John is a small boy. He lives in a village with his brothers and sisters. He goes to school every week.  Could the farmer read it?')
education.loc[education["answer"].str.contains('Yes', na=False), "answer"] = 1
education.loc[education["answer"].str.contains('No', na=False), "answer"] = 0
education = education.rename(columns={'answer':'education'})
education_clean = education.drop(columns = ['question'])
education_clean['education'] = pd.to_numeric(education_clean['education'], errors='coerce')
print(education_clean.head())
print(education_clean['education'].value_counts(dropna= False))

merged_clean = education_clean.merge(surrounding_clean, how = 'outer', left_on= heading, right_on= heading)
merged_clean.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\surrounding&education.csv', index=False)
