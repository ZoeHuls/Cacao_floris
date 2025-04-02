import pyreadr
import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
result = pyreadr.read_r("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\cocoa_survey_dta_sanitized.Rds")
#result_all = pyreadr.read_r("C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\all_farmer_survey_dta_sanitized.Rds")

# Assuming the RData file contains a single data frame, extract it
df = result[None]  # or use the specific key if known, e.g., result['data_frame_name']
#df_all = result_all[None]
# Convert to pandas DataFrame
df = pd.DataFrame(df)
#df_all = pd.DataFrame(df_all)
 
#I know have a file with out olam_farmer_id 
df1 = df[['olam_farmer_id','farmer_id','Gender','country_name','region_name','district_name','place_name','question','answer']]
df1.to_csv('C:\\Users\\zoe.huls\\OneDrive - Olam International\\MsC_Student_Zoe\\MsC_Student_Zoe\\4_Data\\2_Datasets\\cacao_survey_compressed.csv', index=False)