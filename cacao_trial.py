print("Zoeeeeee")
import pyreadr
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
#result = pyreadr.read_r("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\cocoa_survey_dta_sanitized.Rds")
result = pyreadr.read_r("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\all_farmer_survey_dta_sanitized.Rds")

# Assuming the RData file contains a single data frame, extract it
df = result[None]  # or use the specific key if known, e.g., result['data_frame_name']
#df_all = result_all[None]
# Convert to pandas DataFrame
df = pd.DataFrame(df)
#df_all = pd.DataFrame(df_all)
# number of rows to return 
#df = df[['farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']]
#df.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\cocoa_compressed.csv', index=False)
#for chunk in df:
    #categories = ['country_name', 'question']
    #details = chunk[categories]
    #details['count'] = 1
    #summary = details.groupby(categories).sum().reset_index()
    #display(summary.head())
    #break


print(df.head())
#print(df.tail())


# Grouping observations into groups by question looking which question gets asked
#dfg = df.groupby(by = 'question')
#survey_locations = dfg.get_group('Take the current location of survey')
#questions = dfg.groups.keys()
#locations = survey_locations.groupby()
#print(locations)
#survey_locations.describe(include='all')


#location = dfg['Please take the location']
#location = dfg[8]
#location = dfg['Take the current location of survey']
#location.describe(include='all')


# Grouping observations into groups per farmer and extracting a survey of one farmer
#dfg = df.groupby(by = 'farmer_name')
#survey_locations = dfg.get_group('uvkAOyna')
#print(survey_locations)

# Grouping observations into groups per country, extracting all countries
#dfg = df.groupby(by = 'country_name')
#countries = dfg.groups.keys()
#print(countries)

# Grouping observations into groups per farmer_id and extracting a survey of one farmer
#dfg = df.groupby(by = 'farmer_id')
#dfg_all = df_all.groupby(by = 'farmer_id')
#farmer = dfg.get_group('3.123789e-318')
#farmer_all = dfg.get_group('3.123789e-318')
#print(farmer)
#print(farmer_all)
#print("end")

#merging the two datasets
#Large_data = pd.merge(df, df_all, how = "outer", on="farmer_id")
#print(Large_data.head())
#print(Large_data.tail())