"""Construct predictor variables from the combined HRS events dataset.

This script is a direct, ordered extraction of cell 1 from
``database_events_epig_oct.ipynb``. Analytical transformations are intentionally
left unchanged; only path handling and standalone execution are added.
"""

#########################################################################################################
####### CLEANING THE DATASET: REMOVING AND CREATING VARIABLES, SCALING, AGGREGATING, AND RECODING #######
####### input: COMBINED_EVENTS_FILE (master_df)
####### output: CLEAN_EVENTS_FILE (events_df)
####### See data_description.xlsx, sheet "Final_events_db", for more information.
####### last update: 10/06/2025
#########################################################################################################

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from pathlib import Path

# Repository configuration supplies portable input and output paths.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import COMBINED_EVENTS_FILE, CLEAN_EVENTS_FILE

##########################################################################################################
# === 1. Dictionaries of variables ===
##########################################################################################################

            ####### === 1.1 Variables to delete === #########

to_delete_raw = {
    #relationship with parents (not-CLOSE_TIESd)
'LLB032D', 'MLB032D','NLB032D', 'LLB032E', 'MLB032E', 'NLB032E', 
    #life-time major discrimination (not-CLOSE_TIESd)
'KLB036A', 'LLB036A', 'MLB036A', 'NLB036A', 
'KLB036B', 'KLB036C', 'KLB036D', 'KLB036E', 'KLB036F', 'KLB037A', 'KLB037B', 'KLB037C', 'KLB037D', 'KLB037E', 'KLB037F', 'KLB037G', 'KLB037H', 'KLB037I', 'KLB037J', 
'LLB036B', 'LLB036C', 'LLB036D', 'LLB036E', 'LLB036F', 'LLB036G', 'LLB037A', 'LLB037B', 'LLB037C', 'LLB037D', 'LLB037E', 'LLB037F', 'LLB037G', 'LLB037H', 'LLB037I', 'LLB037J', 'LLB037K', 'LLB037M', 'LLB037N', 'LLB037L',
'MLB036B', 'MLB036C', 'MLB036D', 'MLB036E', 'MLB036F', 'MLB036G', 'MLB037A', 'MLB037B', 'MLB037C', 'MLB037D', 'MLB037E', 'MLB037F', 'MLB037G', 'MLB037H', 'MLB037I', 'MLB037J', 'MLB037K', 'MLB037L', 'MLB037M', 'MLB037N',
'NLB036B', 'NLB036C', 'NLB036D', 'NLB036E', 'NLB036F', 'NLB036G', 'NLB037A', 'NLB037B', 'NLB037C', 'NLB037D', 'NLB037E', 'NLB037F', 'NLB037G', 'NLB037H', 'NLB037I', 'NLB037J', 'NLB037K', 'NLB037L', 'NLB037M', 'NLB037N',
    #sexual orientation
'PB135', 
    #childhood health conditions
'MEASLE', 'MUMP', 'CPOX','DIFFSEE', 'ASTHMA','AGEASTHMAD', 'AGEASTHMAE','DIABETES', 'AGEDIABD', 'AGEDIABE','RESP', 'AGERESPD', 'AGERESPE','SPEECH', 'AGESPEECHD', 'AGESPEECHE','ALLERGY', 'AGEALLERGYD', 'AGEALLERGYE','HEART', 'AGEHEARTD', 'AGEHEARTE','EAR', 'AGEEARD', 'AGEEARE','EPILEPSY', 'AGEEPILEPD', 'AGEEPILEPE','MIGRAIN', 'AGEMIGRND', 'AGEMIGRNE', 'STOMACH', 'AGESTOMD', 'AGESTOME','HIGHBP', 'AGEHBPD', 'AGEHBPE', 'DEPRESS', 'AGEDEPRD', 'AGEDEPRE', 'OTPSY', 'AGEPSYCHD', 'AGEPSYCHE', 'CHHEADINJ', 'CHDISABL', 'CHLEARN', 'CHOTHCON',
    #adulthood health conditions
'LH52A', 'LH52B', 'LH52C', 'LH52D', 'LH52E', 'LH52F', 'LH52G', 'LH52H', 'LH52I', 'LH52J', 'LH52K', 'LH52L',
'LH53', 'LH54', 'LH54AM1', 'LH54AM2', 'LH54AM3',    'LH55', 'LH56', 'LH56A', 'LH57','LH57A1A', 'LH57A2A', 'LH57A3A', 'LH57A1B', 'LH57A2B', 'LH57A3B', 'LH57A1C', 'LH57A2C', 'LH57A3C', 
    #childhood learning problems
'LH24', 'LH25','LH26A', 'LH26B', 'LH26C', 'LH26D','LH27A', 'LH27B', 'LH27C', 'LH27D','LH28A', 'LH28B', 'LH28C', 'LH28D','LH29A', 'LH29B', 'LH29C', 'LH29D', 
    #race or ethnicity of the attended school
'LH22_1G', 'LH22_2G', 'LH22_3G', 'LH22_4G', 'LH22_5G',
'LH22_6G', 'LH22_7G', 'LH22_8G', 'LH22_9G', 'LH22_10G',
    #reasons exit job at 30-40
'LH49AM1', 'LH49AM2', 'LH49AM3', 'LH49AM4', 'LH49AM5', 'LH49AM6', 'LH49AM7', 'LH49AM8', 
    #unpaid care variables
'LH51A1B', 'LH51A2B', 'LH51A3B', 'LH51A4B', 'LH51A5B', 'LH51A1C', 'LH51A2C', 'LH51A3C', 'LH51A4C', 'LH51A5C', 
    #position ladder
'OLB036', 'PLB037', 'OLB036', 'PLB037', 
    #major fields of study:
'LH35_1F', 'LH35_2F', 'LH35_3F', 'LH35_4F',  'LH35_5F', 'LH36_1B', 'LH36_2B',  'LH36_3B',  'LH36_4B', 'LH36_5B', 
    #social support other relatives
'OLB011A', 'PLB011A',
    #reasons moved
'PB041M2', 'OB041M2', 'NB041M2', 'MB041M2', 'LB041M2', 'KB041M2',
    #extra
'LH13A', 'LH42B', 
    #highly correlated variables
'RAMEDUC', 'RAFEDUC', 'RAEDYRS', 
    #num children
'PB033', 'OB033', 'NB033', 'MB033', 'LB033', 'KB033'
}

#variables derived that need to be deleted
to_delete_dv1 = {
        #height and weight in feet and lbs
'HEIGHT_FEET', 'HEIGHT_INCHES', 'WEIGHT', 'MAX_WEIGHT', 'REASON_MOVED', 'AGE_STOPPED_SMOKING', 'YEAR_STOPPED_SMOKING', 'PACKS_NOW', 'MAX_PACKS_PER_DAY', 'STILL_COHABITING', 'KNOWNDECEASEDMO', 'KNOWNDECEASEDSOURCE', 'JAIL_TIME_AGG', 'REASON_LIVING_MOTHER', 'NOTWORK_REASON_FAMILY', 'NOTWORK_REASON_HEALTH', 'NOTWORK_REASON_UNEMPLOYED', 'NOTWORK_REASON_NOT_INTERESTED', 'REASON_LIVING_MOTHER', 'LH1A', 'FIRE_FIRED', 'MAIN_OCCUPATION', 'MAIN_INDUSTRY', 'LONGEST_JOB_YEARS', 'CIGS_NOW', 'MAX_CIGS_PER_DAY', 'YEARS_SINCE_STOP_SMOKING', 'REASON_MOVED_CATEGORY', 'PGI_CORTISOL_CORNET14', 'NUM_CAREGIVING_EPISODES',#'JOB_SATISFIED', 'SUPERVISOR_SUPPORT', 'COWORKER_SUPPORT','WORK_ENV', 'JOB_STRESS','NUM_UNI', 'PRIVATE_UNI',
'ALC_GUILT','ALC_CRITICIZED', 'ALC_HANGOVER_DRINK', 'ALC_CUTDOWN', 'NUM_CHILDREN_ALIVE', 'NUM_TIMES_MARRIED',  'GAIN_LOSS_10LBS', 'MILITARY_SERVICE', 'RURAL_CHILDHOOD', 'LH32', 'SIBLINGS_BINARY', 'LIVING_MOTHER', 'AGE_MAX_WEIGHT', 'MAX_WEIGHT_KG', 'HIGHEST_DEGREE', 'LH31H', 'LH31G', 'LH31J','LH31I','LH33A', 'LH17', 'AGE_STARTED_SMOKING_YRS', 'JOB_LOCK_INSURANCE','LH2F','DIVORCED_WIDOWED_SINCE_LAST',  'LH10', 'LH21', 'MARRIED_SINCE_LAST', 'LH4G', 'LH4F','JOB_LOCK_MONEY', 'CURRENT_MARITAL_STATUS', 'SALARY_YEAR', 'LH46M', 'LH45M', 'LH43', 'EXIT_JOB_REASON', 'CURRENTLY_SMOKING', 'BINGE_DAYS_3M', 'BINGE_LIFECOURSE', 'DRINKS_PER_DAY', 'LH48',  'LH49', 'JOB_SAT_30', 'LONGEST_MARRIAGE_YEARS', 'EVER_DISABILITY_DISCRIMINATION', 'EVER_ANCESTRY_DISCRIMINATION', 'EVER_RACE_DISCRIMINATION', 'EVER_RELIGION_DISCRIMINATION', 'EVER_WEIGHT_DISCRIMINATION', 'EVER_AGE_DISCRIMINATION', 'EVER_GENDER_DISCRIMINATION', 'EVER_APPAREANCE_DISCRIMINATION', 'EVER_FINANCIAL_DISCRIMINATION', 'EVER_ORIENTATION_DISCRIMINATION', 'LH31E', 'STILL_MARRIED', 'EVER_WIDOWED',  'EVER_DIVORCED', 'LH31A', 'LH31B', 'LH31C', 'LH31D', 'LH31F', "EVER_MARRIED", "EVER_SMOKED", "EVER_DRANK", 'LH51', 'COWORKER_SUPPORT', 'SUPERVISOR_SUPPORT', 'WORK_ENV', 'JOB_STRESS', 'JOB_SAT_AGG',  'JOB_SATISFIED'
 }


to_delete_dv = {'AGE_MAX_WEIGHT', #66% MV
                'AGE_STARTED_SMOKING_YRS',  #62%
                'AGE_STOPPED_SMOKING',  #98%
                'ALC_CRITICIZED',  #71%
                'ALC_CUTDOWN', #71%
                'ALC_GUILT', #71%
                'ALC_HANGOVER_DRINK', #71%
                'BINGE_DAYS_3M', #43%
                'CIGS_NOW', #83%
                'COWORKER_SUPPORT', #62%
                'CURRENTLY_DRINKS', #USING CURRENTLY_DRINKS_RAND INSTEAD
                'CURRENTLY_SMOKING', #44%
                'CURRENT_MARITAL_STATUS', #55%
                'DIVORCED_WIDOWED_SINCE_LAST', #61%
                'EVER_DIVORCED', #ALREADY IN NUMBER_DIVORCES
                'DAYS_WEEK_DRINKING', #USING RAND VARS INSTEAD,  
                'DRINKS_PER_DAY', #USING RAND VARS INSTEAD
                'EVER_MARRIED', #KEEPING NEVER_MARRIED
                'EVER_WIDOWED', #ALREADY IN NUMBER_WIDOWED
                'EXIT_JOB_REASON', #52%
                'FIRE_FIRED', #83%
                'GAIN_LOSS_10LBS', #67%
                'HEIGHT_FEET', #KEEPING HEIGHT_CM INSTEAD
                'HEIGHT_INCHES', #KEEPING HEIGHT_CM INSTEAD
                'HIGHEST_DEGREE', #65%, USING DEGREE INSTEAD
                'JAIL_TIME_AGG', #92%
                'JOB_LOCK_INSURANCE', #62%
                'JOB_LOCK_MONEY', #52%
                'JOB_SATISFIED', #73%
                'JOB_SAT_30', #33%
                'JOB_SAT_AGG', #56%
                'JOB_STRESS', #55%
                'KC116', 
                'KC134', 
                'KNOWNDECEASEDMO', #82%
                'KNOWNDECEASEDSOURCE', #82%
                'LC116', 
                'LC134', 
                'LH10', #61%
                'LH17', #62%
                'LH1A', #84%
                'LH21', #60%
                'LH2F', #61%
                'LH31G', #64%
                'LH31H', #64%
                'LH31I', #64%
                'LH31J', #64%
                'LH32', #66%
                'LH33A', #64%
                'LH43', #42%
                'LH45M', #42%
                'LH46M', #43%
                'LH48', #39%
                'LH49', #37%
                'LH4F', #59%
                'LH4G', #60%
                'LIVING_MOTHER', #66%
                'LONGEST_JOB_YEARS', #78%
                'LONGEST_MARRIAGE_YEARS', #31%
                'MAIN_INDUSTRY', #78%
                'MAIN_OCCUPATION', #79%
                'MARRIED_SINCE_LAST', #60%
                'MAX_CIGS_PER_DAY', #82%
                'MAX_PACKS_PER_DAY', #91%
                'MAX_WEIGHT', #66%
                'MAX_WEIGHT_KG', #66%
                'MC116', 
                'MILITARY_SERVICE', #66%
                'NC116', 
                'NOTWORK_REASON_FAMILY', #89%
                'NOTWORK_REASON_HEALTH', #89%
                'NOTWORK_REASON_NOT_INTERESTED', #89%
                'NOTWORK_REASON_UNEMPLOYED', #89%
                'NUM_CAREGIVING_EPISODES', #75%
                'NUM_CHILDREN_ALIVE', #71%
                'NUM_TIMES_MARRIED', #70%
                'OC116', 
                'PACKS_NOW', #93%
                'PC116', 
                'PGI_CORTISOL_CORNET14', #75%
                'REASON_LIVING_MOTHER', #89%
                'REASON_MOVED', #74%
                'REASON_MOVED_CATEGORY', #78%
                'RURAL_CHILDHOOD', #66%
                'SALARY_YEAR', #54%
                'STILL_COHABITING', #90%
                'STILL_MARRIED', #USING CURRENTLY_MARRIED INSTEAD
                'SUPERVISOR_SUPPORT', #62%
                'WEIGHT', #USING WEIGHT_KG INSTEAD
                'WORK_ENV', #62%
                'YEARS_SINCE_STOP_SMOKING', #93%
                'YEAR_STOPPED_SMOKING' #97%      
                }

                ####### === 1.2 Variables to scale === #########
network_vars = {
    'LIVES_WITH_PARTNER': ['PLB003', 'OLB003', 'NLB004', 'MLB004', 'LLB004', 'KLB004'], 
    'HAS_CHILDREN': ['PLB006', 'OLB006', 'NLB007', 'MLB007', 'LLB007', 'KLB007'],
    'HAS_FAMILY': ['PLB010', 'OLB010', 'NLB011', 'MLB011', 'LLB011', 'KLB011'],
    'HAS_FRIENDS': ['PLB014', 'OLB014', 'NLB015', 'MLB015', 'LLB015', 'KLB015'],
}

network_vars_1 = {
    'KLB': ['KLB004', 'KLB007', 'KLB011', 'KLB015'],
    'LLB': ['LLB004', 'LLB007', 'LLB011', 'LLB015'],
    'MLB': ['MLB004', 'MLB007', 'MLB011', 'MLB015'],
    'NLB': ['NLB004', 'NLB007', 'NLB011', 'NLB015'],
    'OLB': ['OLB003', 'OLB006', 'OLB010', 'OLB014'],
    'PLB': ['PLB003', 'PLB006', 'PLB010', 'PLB014'],
}

contact_vars = {
    'KLB': {
        'children': ['KLB009A', 'KLB009B', 'KLB009C'],
        'family':   ['KLB013A', 'KLB013B', 'KLB013C'],
        'friends':  ['KLB017A', 'KLB017B', 'KLB017C']
    },
    'LLB': {
        'children': ['LLB009A', 'LLB009B', 'LLB009C'],
        'family':   ['LLB013A', 'LLB013B', 'LLB013C'],
        'friends':  ['LLB017A', 'LLB017B', 'LLB017C']
    },
    'MLB': {
        'children': ['MLB009A', 'MLB009B', 'MLB009C'],
        'family':   ['MLB013A', 'MLB013B', 'MLB013C'],
        'friends':  ['MLB017A', 'MLB017B', 'MLB017C']
    },
    'NLB': {
        'children': ['NLB009A', 'NLB009B', 'NLB009C'],
        'family':   ['NLB013A', 'NLB013B', 'NLB013C'],
        'friends':  ['NLB017A', 'NLB017B', 'NLB017C']
    },
    'OLB': {
        'children': ['OLB008A', 'OLB008B', 'OLB008C', 'OLB008D'],
        'family':   ['OLB012A', 'OLB012B', 'OLB012C', 'OLB012D'],
        'friends':  ['OLB016A', 'OLB016B', 'OLB016C', 'OLB016D']
    },

    'PLB': {
        'children': ['PLB008A', 'PLB008B', 'PLB008C', 'PLB008D'],
        'family':   ['PLB012A', 'PLB012B', 'PLB012C', 'PLB012D'],
        'friends':  ['PLB016A', 'PLB016B', 'PLB016C', 'PLB016D']
    }
}


support_vars = {
    'KLB': {
        'spouse':   ['KLB005A', 'KLB005B', 'KLB005C', 'KLB005D', 'KLB005E', 'KLB005F', 'KLB005G'],
        'children': ['KLB008A', 'KLB008B', 'KLB008C', 'KLB008D', 'KLB008E', 'KLB008F', 'KLB008G'],
        'family':   ['KLB012A', 'KLB012B', 'KLB012C', 'KLB012D', 'KLB012E', 'KLB012F', 'KLB012G'],
        'friends':  ['KLB016A', 'KLB016B', 'KLB016C', 'KLB016D', 'KLB016E', 'KLB016F', 'KLB016G'],
    },
    'LLB': {
        'spouse':   ['LLB005A', 'LLB005B', 'LLB005C', 'LLB005D', 'LLB005E', 'LLB005F', 'LLB005G'],
        'children': ['LLB008A', 'LLB008B', 'LLB008C', 'LLB008D', 'LLB008E', 'LLB008F', 'LLB008G'],
        'family':   ['LLB012A', 'LLB012B', 'LLB012C', 'LLB012D', 'LLB012E', 'LLB012F', 'LLB012G'],
        'friends':  ['LLB016A', 'LLB016B', 'LLB016C', 'LLB016D', 'LLB016E', 'LLB016F', 'LLB016G'],
    },
    'MLB': {
        'spouse':   ['MLB005A', 'MLB005B', 'MLB005C', 'MLB005D', 'MLB005E', 'MLB005F', 'MLB005G'],
        'children': ['MLB008A', 'MLB008B', 'MLB008C', 'MLB008D', 'MLB008E', 'MLB008F', 'MLB008G'],
        'family':   ['MLB012A', 'MLB012B', 'MLB012C', 'MLB012D', 'MLB012E', 'MLB012F', 'MLB012G'],
        'friends':  ['MLB016A', 'MLB016B', 'MLB016C', 'MLB016D', 'MLB016E', 'MLB016F', 'MLB016G'],
    },
    'NLB': {
        'spouse':   ['NLB005A', 'NLB005B', 'NLB005C', 'NLB005D', 'NLB005E', 'NLB005F', 'NLB005G'],
        'children': ['NLB008A', 'NLB008B', 'NLB008C', 'NLB008D', 'NLB008E', 'NLB008F', 'NLB008G'],
        'family':   ['NLB012A', 'NLB012B', 'NLB012C', 'NLB012D', 'NLB012E', 'NLB012F', 'NLB012G'],
        'friends':  ['NLB016A', 'NLB016B', 'NLB016C', 'NLB016D', 'NLB016E', 'NLB016F', 'NLB016G'],
    },
    'OLB': {
        'spouse':   ['OLB004A', 'OLB004B', 'OLB004C', 'OLB004D', 'OLB004E', 'OLB004F', 'OLB004G'],
        'children': ['OLB007A', 'OLB007B', 'OLB007C', 'OLB007D', 'OLB007E', 'OLB007F', 'OLB007G'],
        'family':   ['OLB012A', 'OLB011B', 'OLB011C', 'OLB011D', 'OLB011E', 'OLB011F', 'OLB011G'],
        'friends':  ['OLB015A', 'OLB015B', 'OLB015C', 'OLB015D', 'OLB015E', 'OLB015F', 'OLB015G'],
    },
    'PLB': {
        'spouse':   ['PLB004A', 'PLB004B', 'PLB004C', 'PLB004D', 'PLB004E', 'PLB004F', 'PLB004G'],
        'children': ['PLB007A', 'PLB007B', 'PLB007C', 'PLB007D', 'PLB007E', 'PLB007F', 'PLB007G'],
        'family':   ['PLB012A', 'PLB011B', 'PLB011C', 'PLB011D', 'PLB011E', 'PLB011F', 'PLB011G'],
        'friends':  ['PLB015A', 'PLB015B', 'PLB015C', 'PLB015D', 'PLB015E', 'PLB015F', 'PLB015G'],
    }
}

loneliness_vars = {
    'KLB': {
        'all': ['KLB020A', 'KLB020B', 'KLB020C'],
        'reverse': ['KLB020A', 'KLB020B', 'KLB020C']
    },

    'LLB': {
        'all': ['LLB020A', 'LLB020B', 'LLB020C', 'LLB020D', 'LLB020E', 'LLB020F', 'LLB020G', 'LLB020H', 'LLB020I', 'LLB020J', 'LLB020K'],
        'reverse': ['LLB020A', 'LLB020B', 'LLB020C', 'LLB020E']    
    },

    'MLB': {
        'all': ['MLB020A', 'MLB020B', 'MLB020C', 'MLB020D', 'MLB020E', 'MLB020F', 'MLB020G', 'MLB020H', 'MLB020I', 'MLB020J', 'MLB020K'],
        'reverse': ['MLB020A', 'MLB020B', 'MLB020C', 'MLB020E']   
    },

    'NLB': {
        'all': ['NLB020A', 'NLB020B', 'NLB020C', 'NLB020D', 'NLB020E', 'NLB020F', 'NLB020G', 'NLB020H', 'NLB020I', 'NLB020J', 'NLB020K'],
        'reverse': ['NLB020A', 'NLB020B', 'NLB020C', 'NLB020E']  
    },

    'OLB': {
        'all': ['OLB019A', 'OLB019B', 'OLB019C', 'OLB019D', 'OLB019E', 'OLB019F', 'OLB019G', 'OLB019H', 'OLB019I', 'OLB019J', 'OLB019K'],
        'reverse': ['OLB019A', 'OLB019B', 'OLB019C', 'OLB019E']
    },

    'PLB': {
        'all': ['PLB019A', 'PLB019B', 'PLB019C', 'PLB019D', 'PLB019E', 'PLB019F', 'PLB019G', 'PLB019H', 'PLB019I', 'PLB019J', 'PLB019K'],
        'reverse': ['PLB019A', 'PLB019B', 'PLB019C', 'PLB019E']
    }
}

neighborhood_vars = {
    'KLB': {
        'cohesion': ['KLB021A', 'KLB021C', 'KLB021E', 'KLB021G'],
        'disorder': ['KLB021B', 'KLB021D', 'KLB021F', 'KLB021H'],
        'reverse_disorder': ['KLB021B', 'KLB021D', 'KLB021H']
    },
    'LLB': {
        'cohesion': ['LLB021A', 'LLB021C', 'LLB021E', 'LLB021G'],
        'disorder': ['LLB021B', 'LLB021D', 'LLB021F', 'LLB021H']
    },
    'MLB': {
        'cohesion': ['MLB021A', 'MLB021C', 'MLB021E', 'MLB021G'],
        'disorder': ['MLB021B', 'MLB021D', 'MLB021F', 'MLB021H']
    },
    'NLB': {
        'cohesion': ['NLB021A', 'NLB021C', 'NLB021E', 'NLB021G'],
        'disorder': ['NLB021B', 'NLB021D', 'NLB021F', 'NLB021H']
    },
    'OLB': {
        'cohesion': ['OLB020A', 'OLB020C', 'OLB020E', 'OLB020G'],
        'disorder': ['OLB020B', 'OLB020D', 'OLB020F', 'OLB020H']
    },
    'PLB': {
        'cohesion': ['PLB020A', 'PLB020C', 'PLB020E', 'PLB020G'],
        'disorder': ['PLB020B', 'PLB020D', 'PLB020F', 'PLB020H']
    }
}

discrimination_vars = {
    'KLB': ['KLB030A', 'KLB030B', 'KLB030C', 'KLB030D', 'KLB030E'],
    'LLB': ['LLB030A', 'LLB030B', 'LLB030C', 'LLB030D', 'LLB030E', 'LLB030F'],
    'MLB': ['MLB030A', 'MLB030B', 'MLB030C', 'MLB030D', 'MLB030E', 'MLB030F'],
    'NLB': ['NLB030A', 'NLB030B', 'NLB030C', 'NLB030D', 'NLB030E', 'NLB030F'],
    'OLB': ['OLB029A', 'OLB029B', 'OLB029C', 'OLB029D', 'OLB029E', 'OLB029F'],
    'PLB': ['PLB029A', 'PLB029B', 'PLB029C', 'PLB029D', 'PLB029E', 'PLB029F']
}

reasons_discrimination_vars = {'KLB031M1', 'KLB031M2', 'KLB031M3', 'KLB031M4', 'KLB031M5', 'KLB031M6', 'KLB031M7', 'KLB031M8', 'KLB031M9', 'LLB031M1', 'LLB031M2', 'LLB031M3', 'LLB031M4', 'LLB031M5', 'LLB031M6', 'LLB031M7', 'LLB031M8', 'LLB031M9', 'LLB031M10', 'LLB031M11', 'MLB031M1', 'MLB031M2', 'MLB031M3', 'MLB031M4', 'MLB031M5', 'MLB031M6', 'MLB031M7', 'MLB031M8', 'MLB031M9', 'MLB031M10', 'MLB031M11', 'NLB031M1', 'NLB031M2', 'NLB031M3', 'NLB031M4', 'NLB031M5', 'NLB031M6', 'NLB031M7', 'NLB031M8', 'NLB031M9', 'NLB031M10', 'NLB031M11',
'OLB030M1', 'OLB030M2', 'OLB030M3', 'OLB030M4', 'OLB030M5', 'OLB030M6', 'OLB030M7', 'OLB030M8', 'OLB030M9', 'OLB030M10', 'OLB030M11', 'PLB030M1', 'PLB030M2', 'PLB030M3', 'PLB030M4', 'PLB030M5', 'PLB030M6', 'PLB030M7', 'PLB030M8', 'PLB030M9', 'PLB030M10', 'PLB030M11'}

job_stressors_vars = {
  "KLB": {
    "SAT": {
      "all": ["KLB050A", "KLB050C", "KLB050D", "KLB050E", "KLB050F", "KLB050I", "KLB050J", "KLB050K", "KLB050N"],
      "reverse": ["KLB050E", "KLB050F"]
    },
    "STRESS": {
      "all": ["KLB050B", "KLB050G", "KLB050H", "KLB050L", "KLB050M", "KLB050O"],
      "reverse": []
    }
  },
  "LLB": {
    "SAT": {
      "all": ["LLB050A", "LLB050C", "LLB050D", "LLB050E", "LLB050F", "LLB050I", "LLB050J", "LLB050K", "LLB050N"],
      "reverse": ["LLB050E", "LLB050F"]
    },
    "STRESS": {
      "all": ["LLB050B", "LLB050G", "LLB050H", "LLB050L", "LLB050M", "LLB050O"],
      "reverse": []
    }
  },
  "MLB": {
    "SAT": {
      "all": ["MLB050A", "MLB050C", "MLB050D", "MLB050E", "MLB050F", "MLB050I", "MLB050J", "MLB050K", "MLB050N"],
      "reverse": ["MLB050E", "MLB050F"]
    },
    "STRESS": {
      "all": ["MLB050B", "MLB050G", "MLB050H", "MLB050L", "MLB050M", "MLB050O"],
      "reverse": []
    }
  },
  "NLB": {
    "SAT": {
      "all": ["NLB084A", "NLB084C", "NLB084D", "NLB084E", "NLB084F", "NLB084I", "NLB084J", "NLB084K", "NLB084N"],
      "reverse": ["NLB084E", "NLB084F"]
    },
    "STRESS": {
      "all": ["NLB084B", "NLB084G", "NLB084H", "NLB084L", "NLB084M", "NLB084O"],
      "reverse": []
    }
  }
}

work_env_vars = {
  'LLB': ['LLB050P', 'LLB050Q', 'LLB050R', 'LLB050S', 'LLB050T'],
  'MLB': ['MLB050P', 'MLB050Q', 'MLB050R', 'MLB050S', 'MLB050T'],
  'NLB': ['NLB084P', 'NLB084Q', 'NLB084R', 'NLB084S', 'NLB084T']
}


coworker_support_vars = {
  'LLB': ['LLB050U', 'LLB050V', 'LLB050W'],
  'MLB': ['MLB050U', 'MLB050V', 'MLB050W'],
  'NLB': ['NLB084U', 'NLB084V', 'NLB084W']
}

supervisor_support_vars = {
  'LLB': ['LLB050X', 'LLB050Y', 'LLB050Z', 'LLB050ZA'],
  'MLB': ['MLB050X', 'MLB050Y', 'MLB050Z', 'MLB050ZA'],
  'NLB': ['NLB084X', 'NLB084Y', 'NLB084Z', 'NLB084Z1']
}

hh_income_vars = ['H1ITOT', 'H2ITOT', 'H3ITOT', 'H4ITOT', 'H5ITOT', 'H6ITOT', 'H7ITOT', 'H8ITOT', 'H9ITOT', 'H10ITOT',  'H11ITOT', 'H12ITOT', 'H13ITOT']

unem_vars = ['R1IUNEM', 'R2IUNEM', 'R3IUNEM', 'R4IUNEM', 'R5IUNEM', 'R6IUNEM', 'R7IUNEM', 'R8IUNEM', 'R9IUNEM', 'R10IUNEM', 'R11IUNEM', 'R12IUNEM', 'R13IUNEM']

                ####### === 1.3 Variables to aggregate === #########

close_ties_vars = {
    'CLOSE_CHILDREN' : ['PLB009', 'OLB009', 'NLB010', 'MLB010', 'LLB010', 'KLB010'],
    'CLOSE_FAMILY' : ['PLB013', 'OLB013', 'NLB014', 'MLB014', 'LLB014', 'KLB014'],
    'CLOSE_FRIENDS' : ['PLB017', 'OLB017', 'NLB018', 'MLB018', 'LLB018', 'KLB018']
}

unusual_living_vars = {
    'HOMELESS_AGG': ['NLB035_A', 'OLB033_A'],
    'JAIL_AGG': ['NLB035_B', 'OLB033_B'],
    'JAIL_TIME_AGG': ['NLB035_C', 'OLB033_C']
}

stressful_events_vars = {
    'LOST_JOB_5Y_AGG':['NLB038A', 'MLB038A', 'LLB038A', 'KLB038A'], 
    'UNEMPLOYED_5Y_AGG': ['NLB038B', 'MLB038B', 'LLB038B', 'KLB038B'], 
    'HH_UNEMPLOYED_5Y_AGG': ['NLB038C', 'MLB038C', 'LLB038C', 'KLB038C'], 
    'MOVED_5Y_AGG': ['NLB038D', 'MLB038D', 'LLB038D', 'KLB038D'], 
    'ROBBED_5Y_AGG': ['NLB038E', 'MLB038E', 'LLB038E', 'KLB038E'],
    'FRAUD_5Y_AGG': ['NLB038F', 'MLB038F', 'LLB038F']
} 

ongoing_stressors_vars = {
    'FINANCIAL_STRAIN_ELDERLY': ['PLB035', 'OLB035', 'NLB040', 'MLB040', 'LLB040', 'KLB039B'],
    'HEALTH_PROB_YRSLF': ['PLB035A_1', 'OLB035A_1', 'NLB040A_A', 'MLB040A_A', 'KLB040A'],
    'PHY_EMOT_PROB_FAM': ['PLB035A_2', 'OLB035A_2', 'NLB040A_B', 'MLB040A_B', 'KLB040B'],
    'ALCOHOL_DRUG_FAM': ['PLB035A_3', 'OLB035A_3', 'NLB040A_C', 'MLB040A_C', 'KLB040C'],
    'WORK_DIFF_YRSLF': ['PLB035A_4', 'OLB035A_4', 'NLB040A_D', 'MLB040A_D', 'KLB040D'],
    'FINANCIAL_STRAIN_YRSLF': ['PLB035A_5', 'OLB035A_5', 'NLB040A_E', 'MLB040A_E', 'KLB040E'],
    'HOUSING_PROB_YOURSELF': ['PLB035A_6', 'OLB035A_6', 'NLB040A_F', 'MLB040A_F', 'KLB040F'],
    'RELATIONSHIP_PROB': ['PLB035A_7', 'OLB035A_7', 'NLB040A_G', 'MLB040A_G', 'KLB040G'],
    'SICK_FAM': ['PLB035A_8', 'OLB035A_8', 'NLB040A_H', 'MLB040A_H', 'KLB040H'], 
    'CURRENTLY_WORKING': ['PLB072', 'OLB072', 'MLB045', 'LLB045', 'KLB045'],
    'JOB_LOCK_MONEY': ['PLB073A', 'OLB073A', 'MLB046A', 'LLB046A'],
    'JOB_LOCK_INSURANCE': ['PLB073B', 'OLB073B', 'MLB046B', 'LLB046B'], 
    'NUM_CHILDREN_ALIVE': ['PB034', 'OB034', 'NB034', 'MB034', 'LB034', 'KB034'],
    'REASON_MOVED': ['PB041M1', 'OB041M1', 'NB041M1', 'MB041M1', 'LB041M1', 'KB041M1'], 
    'RELATIONSHIP_SPOUSE': ['PLB005', 'OLB005', 'NLB006', 'MLB006',	'LLB006', 'KLB006'], 
    'RELATIVES_NEIGHBORHOOD': ['PLB013A', 'OLB013A'],
    'FINANCIAL_CONTROL_CHANGE' : ['PLB025A', 'OLB025A', 'NLB026A', 'MLB026A']
}

aggregate_first_valid_vars = {
    'REASON_LIVING_MOTHER': ['PB079', 'OB079', 'NB079', 'MB079', 'LB079', 'KB079'],

    'CURRENT_MARITAL_STATUS': ['PB061', 'OB061', 'NB061', 'MB061', 'LB061', 'KB061'],
    'NUM_TIMES_MARRIED': ['PB065', 'OB065', 'NB065', 'MB065', 'LB065', 'KB065'],
    'MARRIED_SINCE_LAST': ['PB055', 'OB055', 'NB055', 'MB055', 'LB055', 'KB055'],
    'DIVORCED_WIDOWED_SINCE_LAST': ['PB058', 'OB058', 'NB058', 'MB058', 'LB058', 'KB058'],
    'BINGE_LIFECOURSE': ['PC134', 'OC134', 'MC134','NC134'],
    'PHYS_ACTIVITY_VIGOROUS': ['PC223', 'OC223', 'NC223', 'MC223', 'LC223', 'KC223'],
    'PHYS_ACTIVITY_MODERATE': ['PC224', 'OC224', 'NC224', 'MC224', 'LC224', 'KC224'],
    'PHYS_ACTIVITY_MILD': ['PC225', 'OC225', 'NC225', 'MC225', 'LC225', 'KC225'], 
    }

ever_vars_str = {
    'EVER_SMOKED': ['PC116', 'OC116', 'NC116', 'MC116', 'LC116', 'KC116', 'R1SMOKEV', 'R2SMOKEV', 'R3SMOKEV', 'R4SMOKEV', 'R5SMOKEV', 'R6SMOKEV', 'R7SMOKEV', 'R8SMOKEV', 'R9SMOKEV', 'R10SMOKEV', 'R11SMOKEV', 'R12SMOKEV', 'R13SMOKEV'], 
    'EVER_DRANK': ['KC134', 'LC134', 'R1DRINK', 'R2DRINK', 'R3DRINK', 'R4DRINK', 'R5DRINK', 'R6DRINK', 'R7DRINK', 'R8DRINK', 'R9DRINK', 'R10DRINK',  'R11DRINK', 'R12DRINK', 'R13DRINK'],
     }

ever_vars_bin = {
    'LIVING_MOTHER': ['PB078', 'OB078', 'NB078', 'MB078', 'LB078', 'KB078'],
    'FIRE_FIRED': ['PB097', 'OB097', 'NB097', 'MB097', 'LB097'],
    'RURAL_CHILDHOOD': ['PB049', 'OB049', 'NB049', 'MB049', 'LB049', 'KB049'],
    'MILITARY_SERVICE': ['PB035', 'OB035', 'NB035', 'MB035', 'LB035', 'KB035'],
}

health_behavior_vars = {
    # Smoking

    'CURRENTLY_SMOKING': ['PC117', 'OC117', 'NC117', 'MC117', 'LC117', 'KC117'],
    'CIGS_NOW': ['PC118', 'OC118', 'NC118', 'MC118', 'LC118', 'KC118'],
    'PACKS_NOW': ['PC119', 'OC119', 'NC119', 'MC119', 'LC119', 'KC119'],
    'AGE_STARTED_SMOKING_YRS': ['PC120', 'OC120', 'NC120', 'MC120', 'LC120', 'KC120'],
    'AGE_STARTED_SMOKING_YEARS': ['PC121', 'OC121', 'NC121', 'MC121', 'LC121', 'KC121'],
    'AGE_STARTED_SMOKING_AGO': ['PC122', 'OC122', 'NC122', 'MC122', 'LC122', 'KC122'],
    'MAX_CIGS_PER_DAY': ['PC123', 'OC123', 'NC123', 'MC123', 'LC123', 'KC123'],
    'MAX_PACKS_PER_DAY': ['PC124', 'OC124', 'NC124', 'MC124', 'LC124', 'KC124'],
    'YEARS_SINCE_STOP_SMOKING': ['PC125', 'OC125', 'NC125', 'MC125', 'LC125', 'KC125'],
    'YEAR_STOPPED_SMOKING': ['PC126', 'OC126', 'NC126', 'MC126', 'LC126', 'KC126'],
    'AGE_STOPPED_SMOKING': ['PC127', 'OC127', 'NC127', 'MC127', 'LC127', 'KC127'],

    # Alcohol Use
    
    'CURRENTLY_DRINKS': ['PC128', 'OC128', 'NC128', 'MC128', 'LC128', 'KC128'],
    'DAYS_WEEK_DRINKING': ['PC129', 'OC129', 'NC129', 'MC129', 'LC129', 'KC129'],
    'DRINKS_PER_DAY': ['PC130', 'OC130', 'NC130', 'MC130', 'LC130', 'KC130'],
    'BINGE_DAYS_3M': ['PC131', 'OC131', 'NC131', 'MC131', 'LC131', 'KC131'],
    'ALC_CUTDOWN': ['PC135', 'OC135', 'NC135', 'MC135', 'LC135', 'KC135'],
    'ALC_CRITICIZED': ['PC136', 'OC136', 'NC136', 'MC136', 'LC136', 'KC136'],
    'ALC_GUILT': ['PC137', 'OC137', 'NC137', 'MC137', 'LC137', 'KC137'],
    'ALC_HANGOVER_DRINK': ['PC138', 'OC138', 'NC138', 'MC138', 'LC138', 'KC138'],
}


                                #I needed to create three dictionaries because they had different non valid values
antropometric_vars_1 = {
    # Weight
    'WEIGHT': ['PC139', 'OC139', 'NC139', 'MC139', 'LC139', 'KC139'], 
    'MAX_WEIGHT': ['PC226', 'OC226', 'NC226', 'MC226', 'LC226', 'KC226']  
}

antropometric_vars_2 = {    # Height
    'HEIGHT_FEET': ['PC141', 'OC141', 'NC141', 'MC141', 'LC141', 'KC141'],
    'GAIN_LOSS_10LBS': ['PC140', 'OC140', 'NC140', 'MC140', 'LC140', 'KC140']
}

antropometric_vars_3 = {
    'AGE_MAX_WEIGHT': ['PC228', 'OC228', 'NC228', 'MC228', 'LC228', 'KC228'], 
    'HEIGHT_INCHES': ['PC142', 'OC142', 'NC142', 'MC142', 'LC142', 'KC142']
}

job_satisfied_vars = {'JOB_SATISFIED': ['PLB076', 'OLB076']}





                ######### === 1.4 Variables to be derived === ##########

num_educ_institutions_vars = {
    'NUM_SCHOOLS': ['LH22_1A', 'LH22_2A', 'LH22_3A', 'LH22_4A', 'LH22_5A', 'LH22_6A', 'LH22_7A', 'LH22_8A', 'LH22_9A', 'LH22_10A'], 
    'NUM_UNI': ['LH35_1A', 'LH35_2A', 'LH35_3A', 'LH35_4A', 'LH35_5A']
}

private_educ_vars = {
    'PRIVATE_SCHOOL': ['LH22_1F', 'LH22_2F', 'LH22_3F', 'LH22_4F', 'LH22_5F', 'LH22_6F', 'LH22_7F', 'LH22_8F', 'LH22_9F', 'LH22_10F'],
    'PRIVATE_UNI': ['LH35_1D', 'LH35_2D', 'LH35_3D', 'LH35_4D', 'LH35_5D']
}

degree_vars = {
    'HIGHEST_DEGREE': ['LH35_1G', 'LH35_2G', 'LH35_3G', 'LH35_4G', 'LH35_5G']
}

not_work_reason_vars = [
    'LH38AM1', 'LH38AM2', 'LH38AM3', 'LH38AM3', 'LH38AM4', 'LH38AM5', 'LH38AM6'
]

living_at10_vars = ['LH6M1', 'LH6M2', 'LH6M3', 'LH6M4', 'LH6M5', 'LH6M6', 'LH6M7']

lived_at20_vars = ['LH14M1', 'LH14M2', 'LH14M3', 'LH14M4', 'LH14M5', 'LH14M6', 'LH14M7']

lived_at40_vars = ['LH18M1', 'LH18M2', 'LH18M3', 'LH18M4', 'LH18M5', 'LH18M6', 'LH18M7']

relationship_vars = {
    'MARRIAGE': {
        'YEAR_MARRIED': ['LH36_1C', 'LH36_2C', 'LH36_3C', 'LH36_4C', 'LH36_5C'],
        'YEAR_ENDED': ['LH36_1E', 'LH36_2E', 'LH36_3E', 'LH36_4E', 'LH36_5E'],
        'END_STATUS': ['LH36_1D', 'LH36_2D', 'LH36_3D', 'LH36_4D', 'LH36_5D'],
        'EVER_MARRIED_FLAG': 'LH36' 
    },
    'COHABITATION': {
        'YEAR_START': ['LH37_1B', 'LH37_2B', 'LH37_3B', 'LH37_4B', 'LH37_5B'],
        'YEAR_ENDED': ['LH37_1D', 'LH37_2D', 'LH37_3D', 'LH37_4D', 'LH37_5D'],
        'END_STATUS': ['LH37_1C', 'LH37_2C', 'LH37_3C', 'LH37_4C', 'LH37_5C'],
        'EVER_COHAB_FLAG': 'LH37'  
    }
}

job_history_vars = {
    'START_YEAR': ['LH41_1A', 'LH41_2A', 'LH41_3A', 'LH41_4A', 'LH41_5A', 'LH41_6A', 'LH41_7A', 'LH41_8A', 'LH41_9A', 'LH41_10A'],
    'END_YEAR': ['LH41_1B', 'LH41_2B', 'LH41_3B', 'LH41_4B', 'LH41_5B', 'LH41_6B', 'LH41_7B', 'LH41_8B', 'LH41_9B', 'LH41_10B'],
    'OCCUPATION': ['LH41_1FM', 'LH41_2FM', 'LH41_3FM', 'LH41_4FM', 'LH41_5FM', 'LH41_6FM', 'LH41_7FM', 'LH41_8FM', 'LH41_9FM', 'LH41_10FM'],
    'INDUSTRY': ['LH41_1EM', 'LH41_2EM', 'LH41_3EM', 'LH41_4EM', 'LH41_5EM', 'LH41_6EM', 'LH41_7EM', 'LH41_8EM', 'LH41_9EM', 'LH41_10EM'],
    'REASONS': {
            'LH41_1DM1', 'LH41_2DM1', 'LH41_3DM1', 'LH41_4DM1', 'LH41_5DM1', 'LH41_6DM1', 'LH41_7DM1', 'LH41_8DM1', 'LH41_9DM1', 'LH41_10DM1'
            'LH41_1DM2', 'LH41_2DM2', 'LH41_3DM2', 'LH41_4DM2', 'LH41_5DM2', 'LH41_6DM2', 'LH41_7DM2', 'LH41_8DM2', 'LH41_9DM2', 'LH41_10DM2'
            'LH41_1DM3', 'LH41_2DM3', 'LH41_3DM3', 'LH41_4DM3', 'LH41_5DM3', 'LH41_6DM3', 'LH41_7DM3', 'LH41_8DM3', 'LH41_9DM3', 'LH41_10DM3' 'LH41_1DM4', 'LH41_2DM4', 'LH41_3DM4', 'LH41_4DM4', 'LH41_5DM4',
            'LH41_6DM4', 'LH41_7DM4', 'LH41_8DM4', 'LH41_9DM4', 'LH41_10DM4' 
            'LH41_1DM5', 'LH41_2DM5', 'LH41_3DM5', 'LH41_4DM5', 'LH41_5DM5', 'LH41_6DM5', 'LH41_7DM5', 'LH41_8DM5', 'LH41_9DM5', 'LH41_10DM5'
            'LH41_1DM6', 'LH41_2DM6', 'LH41_3DM6', 'LH41_4DM6', 'LH41_5DM6',  'LH41_6DM6', 'LH41_7DM6', 'LH41_8DM6', 'LH41_9DM6', 'LH41_10DM6'
        
    }
}

unpaid_care_vars = {
    'relationship_vars': ['LH51A1A', 'LH51A2A', 'LH51A3A', 'LH51A4A', 'LH51A5A'],
    'start_vars': ['LH51A1B', 'LH51A2B', 'LH51A3B', 'LH51A4B', 'LH51A5B'],
    'end_vars': ['LH51A1C', 'LH51A2C', 'LH51A3C', 'LH51A4C', 'LH51A5C']

}



                ######### === 1.5 Derived variables (from event_df) to be aggregated === ##########

scores_to_aggregate_vars = {
    'CONTACT_CHILDREN': [
        'PLB_contact_children', 'OLB_contact_children', 'NLB_contact_children',
        'MLB_contact_children', 'LLB_contact_children', 'KLB_contact_children'
    ],

    'CONTACT_FAMILY': [
        'PLB_contact_family', 'OLB_contact_family', 'NLB_contact_family',
        'MLB_contact_family', 'LLB_contact_family', 'KLB_contact_family'
    ],

    'CONTACT_FRIENDS': [
        'PLB_contact_friends', 'OLB_contact_friends', 'NLB_contact_friends',
        'MLB_contact_friends', 'LLB_contact_friends', 'KLB_contact_friends'
    ],

    'SPOUSE_POS_SUPPORT': [
        'PLB_spouse_pos_support', 'OLB_spouse_pos_support', 'NLB_spouse_pos_support',
        'MLB_spouse_pos_support', 'LLB_spouse_pos_support', 'KLB_spouse_pos_support'
    ],

    'CHILDREN_POS_SUPPORT': [
        'PLB_children_pos_support', 'OLB_children_pos_support', 'NLB_children_pos_support',
        'MLB_children_pos_support', 'LLB_children_pos_support', 'KLB_children_pos_support'
    ],

    'FAMILY_POS_SUPPORT': [
        'PLB_family_pos_support', 'OLB_family_pos_support', 'NLB_family_pos_support',
        'MLB_family_pos_support', 'LLB_family_pos_support', 'KLB_family_pos_support'
    ],

    'FRIENDS_POS_SUPPORT': [
        'PLB_friends_pos_support', 'OLB_friends_pos_support', 'NLB_friends_pos_support',
        'MLB_friends_pos_support', 'LLB_friends_pos_support', 'KLB_friends_pos_support'
    ],

    'SPOUSE_NEG_SUPPORT': [
        'PLB_spouse_neg_support', 'OLB_spouse_neg_support', 'NLB_spouse_neg_support',
        'MLB_spouse_neg_support', 'LLB_spouse_neg_support', 'KLB_spouse_neg_support'
    ],

    'CHILDREN_NEG_SUPPORT': [
        'PLB_children_neg_support', 'OLB_children_neg_support', 'NLB_children_neg_support',
        'MLB_children_neg_support', 'LLB_children_neg_support', 'KLB_children_neg_support'
    ],

    'FAMILY_NEG_SUPPORT': [
        'PLB_family_neg_support', 'OLB_family_neg_support', 'NLB_family_neg_support',
        'MLB_family_neg_support', 'LLB_family_neg_support', 'KLB_family_neg_support'
    ],

    'FRIENDS_NEG_SUPPORT': [
        'PLB_friends_neg_support', 'OLB_friends_neg_support', 'NLB_friends_neg_support',
        'MLB_friends_neg_support', 'LLB_friends_neg_support', 'KLB_friends_neg_support'
    ],

    'LONELINESS': [
        'PLB_loneliness', 'OLB_loneliness', 'NLB_loneliness',
        'MLB_loneliness', 'LLB_loneliness', 'KLB_loneliness'
    ],

    'NEIGHBORHOOD_COHESION': [
        'PLB_neigh_cohesion', 'OLB_neigh_cohesion', 'NLB_neigh_cohesion',
        'MLB_neigh_cohesion', 'LLB_neigh_cohesion', 'KLB_neigh_cohesion'
    ],

    'NEIGHBORHOOD_DISORDER': [
        'PLB_neigh_disorder', 'OLB_neigh_disorder', 'NLB_neigh_disorder',
        'MLB_neigh_disorder', 'LLB_neigh_disorder', 'KLB_neigh_disorder'
    ],

    'DISCRIMINATION': [
        'PLB_discrimination', 'OLB_discrimination', 'NLB_discrimination',
        'MLB_discrimination', 'LLB_discrimination', 'KLB_discrimination'
    ], 
    'JOB_SAT_AGG': [
        'NLB_JOB_SAT', 'MLB_JOB_SAT', 'LLB_JOB_SAT', 'KLB_JOB_SAT'],

    'JOB_STRESS': [
        'NLB_JOB_STRESS', 'MLB_JOB_STRESS', 'LLB_JOB_STRESS', 'KLB_JOB_STRESS'],

    'WORK_ENV': [
        'NLB_WORK_ENV', 'MLB_WORK_ENV', 'LLB_WORK_ENV'],

    'COWORKER_SUPPORT': [
       'NLB_COWORKER_SUPPORT', 'MLB_COWORKER_SUPPORT', 'LLB_COWORKER_SUPPORT'],

    'SUPERVISOR_SUPPORT': [
        'NLB_SUPERVISOR_SUPPORT', 'MLB_SUPERVISOR_SUPPORT', 'LLB_SUPERVISOR_SUPPORT']
}


pgs_vars = {
    'AFBC_SOCGEN16': ['A5_AFBC_SOCGEN16', 'E5_AFBC_SOCGEN16', 'H5_AFBC_SOCGEN16'],
    'MENARCHE_REPROGEN17': ['A5_MENARCHE_REPROGEN17', 'E5_MENARCHE_REPROGEN17', 'H5_MENARCHE_REPROGEN17'],
    'MENOPAUSEREPROGEN21': ['A5_MENOPAUSEREPROGEN21', 'E5_MENOPAUSEREPROGEN21', 'H5_MENOPAUSEREPROGEN21'],
    'AI_GSCAN19': ['A5_AI_GSCAN19', 'E5_AI_GSCAN19', 'H5_AI_GSCAN19'],
    'DPW_GSCAN19': ['A5_DPW_GSCAN19', 'E5_DPW_GSCAN19', 'H5_DPW_GSCAN19'],
    'GENCOG2_CHARGE18': ['A5_GENCOG2_CHARGE18', 'E5_GENCOG2_CHARGE18', 'H5_GENCOG2_CHARGE18'],
    'CANNABIS_ICC18': ['A5_CANNABIS_ICC18', 'E5_CANNABIS_ICC18', 'H5_CANNABIS_ICC18'],
    'NEBC_SOCGEN16': ['A5_NEBC_SOCGEN16', 'E5_NEBC_SOCGEN16', 'H5_NEBC_SOCGEN16'],
    'SC_GSCAN19': ['A5_SC_GSCAN19', 'E5_SC_GSCAN19', 'H5_SC_GSCAN19'],
    'SI_GSCAN19': ['A5_SI_GSCAN19', 'E5_SI_GSCAN19', 'H5_SI_GSCAN19'],
    'WC_GIANT15': ['A5_WC_GIANT15', 'E5_WC_GIANT15', 'H5_WC_GIANT15'],
    'WHR_GIANT15': ['A5_WHR_GIANT15', 'E5_WHR_GIANT15', 'H5_WHR_GIANT15'],
    'CORTISOL_CORNET14': ['A5_CORTISOL_CORNET14', 'H5_CORTISOL_CORNET14'],
    'HTN_COGNET17': ['A5_HTN_COGNET17', 'E5_HTN_COGNET17', 'H5_HTN_COGNET17'],
    'BMI2_GIANT18': ['A5_BMI2_GIANT18', 'E5_BMI2_GIANT18', 'H5_BMI2_GIANT18'],
    'CRP_CHARGE22': ['A5_CRP_CHARGE22', 'E5_CRP_CHARGE22', 'H5_CRP_CHARGE22'],
    'EDU3_W23_SSGAC18': ['A5_EDU3_W23_SSGAC18', 'E5_EDU3_W23_SSGAC18', 'H5_EDU3_W23_SSGAC18'],
    'EXTRAVERSION_GPC16': ['A5_EXTRAVERSION_GPC16', 'E5_EXTRAVERSION_GPC16', 'H5_EXTRAVERSION_GPC16'],
    'HBA1CEA_MAGIC17': ['A5_HBA1CEA_MAGIC17', 'E5_HBA1CEA_MAGIC17', 'H5_HBA1CEA_MAGIC17'],
    'BUN_CKDGEN19': ['A5_BUN_CKDGEN19', 'E5_BUN_CKDGEN19', 'H5_BUN_CKDGEN19'],
    'BUNTE_CKDGEN19': ['A5_BUNTE_CKDGEN19', 'E5_BUNTE_CKDGEN19', 'H5_BUNTE_CKDGEN19'],
    'CKDTE_CKDGEN19': ['A5_CKDTE_CKDGEN19', 'E5_CKDTE_CKDGEN19', 'H5_CKDTE_CKDGEN19'],
    'CKD_CKDGEN19': ['A5_CKD_CKDGEN19', 'E5_CKD_CKDGEN19', 'H5_CKD_CKDGEN19'],
    'EGFRTE_CKDGEN19': ['A5_EGFRTE_CKDGEN19', 'E5_EGFRTE_CKDGEN19', 'H5_EGFRTE_CKDGEN19'],
    'EGFR_CKDGEN19': ['A5_EGFR_CKDGEN19', 'E5_EGFR_CKDGEN19', 'H5_EGFR_CKDGEN19'],
    'HDL_GLGC13': ['A5_HDL_GLGC13', 'E5_HDL_GLGC13', 'H5_HDL_GLGC13'],
    'LDL_GLGC13': ['A5_LDL_GLGC13', 'E5_LDL_GLGC13', 'H5_LDL_GLGC13'],
    'TC_GLGC13': ['A5_TC_GLGC13', 'E5_TC_GLGC13', 'H5_TC_GLGC13'],
    'TG_GLGC13': ['A5_TG_GLGC13', 'E5_TG_GLGC13', 'H5_TG_GLGC13'],
    'NEUROTICISM_SSGAC16': ['A5_NEUROTICISM_SSGAC16', 'E5_NEUROTICISM_SSGAC16', 'H5_NEUROTICISM_SSGAC16'],
    'WELLBEING_SSGAC16': ['A5_WELLBEING_SSGAC16', 'E5_WELLBEING_SSGAC16', 'H5_WELLBEING_SSGAC16'],
    'ALC_PGC18': ['A5_ALC_PGC18', 'E5_ALC_PGC18', 'H5_ALC_PGC18'],
    'GWALZNA_PGC21': ['A5_GWALZNA_PGC21', 'E5_GWALZNA_PGC21', 'H5_GWALZNA_PGC21'],
    'AB_BROAD17': ['A5_AB_BROAD17', 'E5_AB_BROAD17', 'H5_AB_BROAD17'],
    'ANXFS_ANGST16': ['A5_ANXFS_ANGST16', 'E5_ANXFS_ANGST16', 'H5_ANXFS_ANGST16'],
    'ADHD_PGC17': ['A5_ADHD_PGC17', 'E5_ADHD_PGC17', 'H5_ADHD_PGC17'],
    'AUTISM_PGC17': ['A5_AUTISM_PGC17', 'E5_AUTISM_PGC17', 'H5_AUTISM_PGC17'],
    'BIP_PGC11': ['A5_BIP_PGC11', 'E5_BIP_PGC11', 'H5_BIP_PGC11'],
    'CAD_CARDIOGRAM11': ['A5_CAD_CARDIOGRAM11', 'E5_CAD_CARDIOGRAM11', 'H5_CAD_CARDIOGRAM11'],
    'DEPSYMP_SSGAC16': ['A5_DEPSYMP_SSGAC16', 'E5_DEPSYMP_SSGAC16', 'H5_DEPSYMP_SSGAC16'],
    'MI_CARDIOGRAM15': ['A5_MI_CARDIOGRAM15', 'E5_MI_CARDIOGRAM15', 'H5_MI_CARDIOGRAM15'],
    'XDISORDER_PGC13': ['A5_XDISORDER_PGC13', 'E5_XDISORDER_PGC13', 'H5_XDISORDER_PGC13'],
    'OCD_IOCDF17': ['A5_OCD_IOCDF17', 'E5_OCD_IOCDF17', 'H5_OCD_IOCDF17'],
    'PTSDC_PGC18': ['A5_PTSDC_PGC18', 'E5_PTSDC_PGC18', 'H5_PTSDC_PGC18'],
    'SCZ_PGC14': ['A5_SCZ_PGC14', 'E5_SCZ_PGC14', 'H5_SCZ_PGC14'],
    'T2DALL_DIAGRAM24': ['A5_T2DALL_DIAGRAM24', 'E5_T2DALL_DIAGRAM24', 'H5_T2DALL_DIAGRAM24']

}

sibling_binary_cols = ['KB080', 'LB080', 'MB080', 'NB080', 'OB080', 'PB080']
sibling_count_cols = [
    'R1LIVSIB', 'R2LIVSIB', 'R3LIVSIB', 'R4LIVSIB', 'R5LIVSIB', 'R6LIVSIB',
    'R7LIVSIB', 'R8LIVSIB', 'R9LIVSIB', 'R10LIVSIB', 'R11LIVSIB', 'R12LIVSIB', 'R13LIVSIB'
]

siblings_vars = {
    'SIBLINGS_BINARY': sibling_binary_cols + sibling_count_cols
}

rand_vars = {
    'NUMBER_DIVORCES': ['R13MDIV', 'R12MDIV', 'R11MDIV', 'R10MDIV', 'R9MDIV', 'R8MDIV', 'R7MDIV', 'R6MDIV', 'R5MDIV', 'R4MDIV', 'R3MDIV', 'R2MDIV', 'R1MDIV'], 

    'NUMBER_MARRIAGES' : ['R13MRCT', 'R12MRCT', 'R11MRCT', 'R10MRCT', 'R9MRCT', 'R8MRCT','R7MRCT', 'R6MRCT', 'R5MRCT', 'R4MRCT', 'R3MRCT', 'R2MRCT', 'R1MRCT'], 

    'NEVER_MARRIED' : ['R13MNEV', 'R12MNEV', 'R11MNEV', 'R10MNEV', 'R9MNEV', 'R8MNEV', 'R7MNEV', 'R6MNEV', 'R5MNEV', 'R4MNEV', 'R3MNEV', 'R2MNEV', 'R1MNEV'], 

    'NUMBER_WIDOWED' : ['R13MWID', 'R12MWID', 'R11MWID', 'R10MWID', 'R9MWID', 'R8MWID', 'R7MWID', 'R6MWID', 'R5MWID', 'R4MWID', 'R3MWID', 'R2MWID', 'R1MWID'], 

    'CURRENTLY_SMOKING_RAND' : ['R13SMOKEN', 'R12SMOKEN', 'R11SMOKEN', 'R10SMOKEN', 'R9SMOKEN', 'R8SMOKEN', 'R7SMOKEN', 'R6SMOKEN', 'R5SMOKEN', 'R4SMOKEN', 'R3SMOKEN', 'R2SMOKEN', 'R1SMOKEN'], 

    'EVER_SMOKED_RAND' : ['R13SMOKEV', 'R12SMOKEV', 'R11SMOKEV', 'R10SMOKEV', 'R9SMOKEV', 'R8SMOKEV', 'R7SMOKEV', 'R6SMOKEV', 'R5SMOKEV', 'R4SMOKEV', 'R3SMOKEV', 'R2SMOKEV', 'R1SMOKEV'], 

    'NUMBER_LIVING_SIBLINGS': ['R13LIVSIB', 'R12LIVSIB', 'R11LIVSIB', 'R10LIVSIB', 'R9LIVSIB', 'R8LIVSIB', 'R7LIVSIB', 'R6LIVSIB', 'R5LIVSIB', 'R4LIVSIB', 'R3LIVSIB', 'R2LIVSIB', 'R1LIVSIB'], 

    'MARITAL_STATUS_RAND':  ['R13MSTAT', 'R12MSTAT', 'R11MSTAT', 'R10MSTAT', 'R9MSTAT', 'R8MSTAT', 'R7MSTAT', 'R6MSTAT', 'R5MSTAT', 'R4MSTAT', 'R3MSTAT', 'R2MSTAT', 'R1MSTAT'], 

    'CURRENTLY_DRINKS_RAND': ['R13DRINK', 'R12DRINK', 'R11DRINK', 'R10DRINK', 'R9DRINK', 'R8DRINK', 'R7DRINK', 'R6DRINK', 'R5DRINK', 'R4DRINK', 'R3DRINK', 'R2DRINK', 'R1DRINK'], 

    'DRINKS_PER_DAY_RAND': ['R13DRINKN', 'R12DRINKN', 'R11DRINKN', 'R10DRINKN', 'R9DRINKN', 'R8DRINKN', 'R7DRINKN', 'R6DRINKN', 'R5DRINKN', 'R4DRINKN', 'R3DRINKN'],

    'HH_POVERTY_THRESHOLD': ['H13INPOV',  'H12INPOV', 'H11INPOV', 'H10INPOV', 'H9INPOV', 'H8INPOV', 'H7INPOV', 'H6INPOV'],

    'CURRENTLY_PENSION': ['R13PENINC', 'R12PENINC', 'R11PENINC', 'R10PENINC', 'R9PENINC', 'R8PENINC', 'R7PENINC', 'R6PENINC', 'R5PENINC', 'R4PENINC', 'R3PENINC', 'R2PENINC'],

    'LONGEST_JOB_OCCUPATION': ['R13JLOCC', 'R12JLOCC', 'R11JLOCC', 'R10JLOCC', 'R9JLOCC', 'R8JLOCC', 'R7JLOCC', 'R6JLOCC', 'R5JLOCC', 'R4JLOCC', 'R3JLOCC', 'R2JLOCC', 'R1JLOCC'], 

    'LONGEST_JOB_INDUSTRY': ['R13JLIND', 'R12JLIND', 'R11JLIND', 'R10JLIND', 'R9JLIND', 'R8JLIND', 'R7JLIND', 'R6JLIND', 'R5JLIND', 'R4JLIND', 'R3JLIND', 'R2JLIND', 'R1JLIND'], 

    'LONGEST_JOB_DURATION': ['R13JLTEN', 'R12JLTEN', 'R11JLTEN', 'R10JLTEN', 'R9JLTEN', 'R8JLTEN', 'R7JLTEN', 'R6JLTEN', 'R5JLTEN', 'R4JLTEN', 'R3JLTEN', 'R2JLTEN', 'R1JLTEN'],

    'DAYS_WEEK_DRINKING_RAND' : ['R13DRINKD', 'R12DRINKD', 'R11DRINKD', 'R10DRINKD', 'R9DRINKD', 'R8DRINKD', 'R7DRINKD', 'R6DRINKD', 'R5DRINKD', 'R4DRINKD', 'R3DRINKD'],

    'DAD_ALIVE_RAND': ['R13DADLIV', 'R12DADLIV', 'R11DADLIV', 'R10DADLIV', 'R9DADLIV', 'R8DADLIV', 'R7DADLIV', 'R6DADLIV', 'R5DADLIV', 'R4DADLIV', 'R3DADLIV', 'R2DADLIV', 'R1DADLIV'],

    'MOTHER_ALIVE_RAND': ['R13MOMLIV', 'R12MOMLIV', 'R11MOMLIV', 'R10MOMLIV', 'R9MOMLIV', 'R8MOMLIV', 'R7MOMLIV', 'R6MOMLIV', 'R5MOMLIV', 'R4MOMLIV', 'R3MOMLIV', 'R2MOMLIV', 'R1MOMLIV'],

    'MOTHER_AGE_RAND': ['R13MOMAGE', 'R12MOMAGE', 'R11MOMAGE', 'R10MOMAGE', 'R9MOMAGE', 'R8MOMAGE', 'R7MOMAGE', 'R6MOMAGE', 'R5MOMAGE', 'R4MOMAGE', 'R3MOMAGE', 'R2MOMAGE', 'R1MOMAGE'],

    'DAD_AGE_RAND': ['R13DADAGE', 'R12DADAGE', 'R11DADAGE', 'R10DADAGE', 'R9DADAGE', 'R8DADAGE', 'R7DADAGE', 'R6DADAGE', 'R5DADAGE', 'R4DADAGE', 'R3DADAGE', 'R2DADAGE', 'R1DADAGE'],

    'LONGEST_MARRIAGE_YRS_RAND': ['R13MLEN', 'R12MLEN', 'R11MLEN', 'R10MLEN', 'R9MLEN', 'R8MLEN', 'R7MLEN', 'R6MLEN', 'R5MLEN', 'R4MLEN', 'R3MLEN', 'R2MLEN', 'R1MLEN'],

    'CURRENT_MARRIAGE_YRS_RAND': ['R13MCURLN', 'R12MCURLN', 'R11MCURLN', 'R10MCURLN', 'R9MCURLN', 'R8MCURLN', 'R7MCURLN', 'R6MCURLN', 'R5MCURLN', 'R4MCURLN', 'R3MCURLN', 'R2MCURLN', 'R1MCURLN'],
    
    'RELIGION_RAND': ['RARELIG'],
    'VETERAN_RAND': ['RAVETRN'],
    'BIRTH_PLACE_RAND': ['RABPLACE']
}

rand_vars_str = [
    #MARITAL_STATUS_RAND
    'R13MSTAT', 'R12MSTAT', 'R11MSTAT', 'R10MSTAT', 'R9MSTAT', 'R8MSTAT', 'R7MSTAT', 'R6MSTAT', 'R5MSTAT', 'R4MSTAT', 'R3MSTAT', 'R2MSTAT', 'R1MSTAT', 
    #EVER_DRANK_RAND
    'R13DRINK', 'R12DRINK', 'R11DRINK', 'R10DRINK', 'R9DRINK', 'R8DRINK', 'R7DRINK', 'R6DRINK', 'R5DRINK', 'R4DRINK', 'R3DRINK', 'R2DRINK', 'R1DRINK',
    #DAYS_WEEK_DRINKING_RAND
    'R3DRINKD', 'R4DRINKD', 'R5DRINKD', 'R6DRINKD', 'R7DRINKD', 'R8DRINKD', 'R9DRINKD', 'R10DRINKD', 'R11DRINKD', 'R12DRINKD', 'R13DRINKD',
    #CURRENTLY_SMOKING_RAND
    'R13SMOKEN', 'R12SMOKEN', 'R11SMOKEN', 'R10SMOKEN', 'R9SMOKEN', 'R8SMOKEN', 'R7SMOKEN', 'R6SMOKEN', 'R5SMOKEN', 'R4SMOKEN', 'R3SMOKEN', 'R2SMOKEN', 'R1SMOKEN',
    #EVER_SMOKED_RAND
    'R13SMOKEV', 'R12SMOKEV', 'R11SMOKEV', 'R10SMOKEV', 'R9SMOKEV', 'R8SMOKEV', 'R7SMOKEV', 'R6SMOKEV', 'R5SMOKEV', 'R4SMOKEV', 'R3SMOKEV', 'R2SMOKEV', 'R1SMOKEV',
    #NEVER_MARRIED
    'R13MNEV', 'R12MNEV', 'R11MNEV', 'R10MNEV', 'R9MNEV', 'R8MNEV', 'R7MNEV', 'R6MNEV', 'R5MNEV', 'R4MNEV', 'R3MNEV', 'R2MNEV', 'R1MNEV',
    #HH_POVERTY_THRESHOLD
    'H6INPOV', 'H7INPOV', 'H8INPOV', 'H9INPOV', 'H10INPOV', 'H11INPOV', 'H12INPOV', 'H13INPOV',
    #CURRENTLY_PENSION
    'R2PENINC', 'R3PENINC', 'R4PENINC', 'R5PENINC', 'R6PENINC', 'R7PENINC', 'R8PENINC', 'R9PENINC', 'R10PENINC', 'R11PENINC', 'R12PENINC', 'R13PENINC', 
    #LONGEST_JOB_OCCUPATION
    'R1JLOCC', 'R2JLOCC', 'R3JLOCC', 'R4JLOCC', 'R5JLOCC', 'R6JLOCC', 'R7JLOCC', 'R8JLOCC', 'R9JLOCC', 'R10JLOCC', 'R11JLOCC', 'R12JLOCC','R13JLOCC', 
    #LONGEST_JOB_INDUSTRY
    'R1JLIND', 'R2JLIND', 'R3JLIND', 'R4JLIND', 'R5JLIND', 'R6JLIND', 'R7JLIND', 'R8JLIND', 'R9JLIND', 'R10JLIND', 'R11JLIND', 'R12JLIND', 'R13JLIND',
    #NUMBER_DRINKS
    'R13DRINKN', 'R12DRINKN', 'R11DRINKN', 'R10DRINKN', 'R9DRINKN', 'R8DRINKN', 'R7DRINKN', 'R6DRINKN', 'R5DRINKN', 'R4DRINKN', 'R3DRINKN', 
    #EVER_SMOKED_RAND
    'R1SMOKEV', 'R2SMOKEV', 'R3SMOKEV', 'R4SMOKEV', 'R5SMOKEV', 'R6SMOKEV', 'R7SMOKEV', 'R8SMOKEV', 'R9SMOKEV', 'R10SMOKEV', 'R11SMOKEV', 'R12SMOKEV', 'R13SMOKEV',  
    'RAEDUC', 
    #MOM AND FATHER ALIVE
    'R1MOMLIV', 'R2MOMLIV', 'R3MOMLIV', 'R4MOMLIV', 'R5MOMLIV', 'R6MOMLIV', 'R7MOMLIV', 'R8MOMLIV', 'R9MOMLIV', 'R10MOMLIV', 'R11MOMLIV', 'R12MOMLIV', 'R13MOMLIV',
    'R1DADLIV', 'R2DADLIV', 'R3DADLIV', 'R4DADLIV', 'R5DADLIV', 'R6DADLIV', 'R7DADLIV', 'R8DADLIV', 'R9DADLIV', 'R10DADLIV', 'R11DADLIV', 'R12DADLIV', 'R13DADLIV',
    #RELIGION
    'RARELIG', 'RAVETRN', 'RABPLACE'
]
















##########################################################################################################
# === 2. Functions ===
##########################################################################################################


def compute_network_composition(df, network_vars):
    result = pd.DataFrame(index=df.index)
    for wave_prefix, vars_list in network_vars.items():
        valid_mask = df[vars_list].notna().all(axis=1) # Boolean mask for complete rows
        composition_score = df[vars_list].apply(lambda x: (x == 1).sum(), axis=1) # Count how many of the 4 variables equal 1 (only for complete rows)
        composition_score[~valid_mask] = pd.NA #If any of the variables is missing it becomes a missing value
        result[f"{wave_prefix}_network_composition"] = composition_score
    return result, [v for varlist in network_vars.values() for v in varlist]



def compute_contact_frequency(df, contact_vars):
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave_prefix, categories in contact_vars.items():
        for group_name, var_list in categories.items():
            # Save the used variables
            used_vars.extend(var_list)

            # Subset and check missing
            data = df[var_list]
            valid_mask = data.notna().all(axis=1)

            # Reverse code (1=frequent → becomes 6)
            reversed_data = data.apply(lambda x: 7 - x)

            # Compute mean only for complete cases
            mean_contact = reversed_data.mean(axis=1)
            mean_contact[~valid_mask] = pd.NA

            # Store in results
            result[f'{wave_prefix}_contact_{group_name}'] = mean_contact

    return result, used_vars

def compute_perceived_social_support(df, support_vars):
    """
    Compute positive and negative social support scores for each wave and relationship group.
    Applies reverse coding and missing data rules:
    - Positive support (a–c): set to NaN if >1 item missing
    - Negative support (d–g): set to NaN if >2 items missing
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave_prefix, groups in support_vars.items():
        for group, items in groups.items():
            used_vars.extend(items)

            # Split items: first 3 = positive, last 4 = negative
            pos_items = items[:3]
            neg_items = items[3:]

            #reverse coding
            pos_data = df[pos_items].apply(lambda x: 5 - x)
            neg_data = df[neg_items].apply(lambda x: 5 - x)

            #counting missing values
            pos_missing = df[pos_items].isna().sum(axis=1)
            neg_missing = df[neg_items].isna().sum(axis=1)

            #final score: mean of the values of each item, dividing positive and negative support 
            pos_score = pos_data.mean(axis=1)
            neg_score = neg_data.mean(axis=1)

            #setting as NA observations with more than 1 (positive) or 2 (negative) missing values
            pos_score[pos_missing > 1] = pd.NA
            neg_score[neg_missing > 2] = pd.NA

            result[f"{wave_prefix}_{group}_pos_support"] = pos_score
            result[f"{wave_prefix}_{group}_neg_support"] = neg_score

    return result, used_vars


def compute_loneliness_index(df, loneliness_vars):
    """
    Compute loneliness index for each wave:
    - KLB: uses only 3 items (A–C), missing if >1 missing
    - Other waves: use all 11 items, missing if >5 missing
    - Items A, B, C, E are reverse-coded
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave_prefix, info in loneliness_vars.items():
        all_items = info['all']
        reverse_items = info['reverse']
        used_vars.extend(all_items)

        # Copy and reverse code A, B, C, E
        data = df[all_items].copy()
        data[reverse_items] = data[reverse_items].apply(lambda x: 4 - x)

        if wave_prefix == 'KLB':
            # Only 3-item version
            data_3 = data[reverse_items[:3]]
            missing_3 = data_3.isna().sum(axis=1)
            score = data_3.mean(axis=1)
            score[missing_3 > 1] = pd.NA
        else:
            # Full 11-item version
            missing_11 = data.isna().sum(axis=1)
            score = data.mean(axis=1)
            score[missing_11 > 5] = pd.NA

        result[f"{wave_prefix}_loneliness"] = score

    return result, used_vars

def compute_neighborhood_scales(df, neighborhood_vars):
    """
    Computes neighborhood disorder and cohesion scales.
    
    Rules:
    - Cohesion: all items are reverse-coded; NaN if >2 missing
    - Disorder: reverse-code 3 items in KLB only (B, D, H); NaN if >2 missing
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, items in neighborhood_vars.items():
        cohesion_items = items['cohesion']
        disorder_items = items['disorder']
        reverse_disorder = items.get('reverse_disorder', [])

        # Track used variables
        used_vars.extend(cohesion_items + disorder_items)

        # --- Cohesion ---
        cohesion_data = df[cohesion_items].copy()
        cohesion_data = cohesion_data.apply(lambda x: 8 - x)  # Reverse-code all
        missing_cohesion = cohesion_data.isna().sum(axis=1)
        cohesion_score = cohesion_data.mean(axis=1)
        cohesion_score[missing_cohesion > 2] = pd.NA
        result[f"{wave}_neigh_cohesion"] = cohesion_score

        # --- Disorder ---
        disorder_data = df[disorder_items].copy()
        if reverse_disorder:
            disorder_data[reverse_disorder] = disorder_data[reverse_disorder].apply(lambda x: 8 - x)
        missing_disorder = disorder_data.isna().sum(axis=1)
        disorder_score = disorder_data.mean(axis=1)
        disorder_score[missing_disorder > 2] = pd.NA
        result[f"{wave}_neigh_disorder"] = disorder_score

    return result, used_vars

def compute_discrimination_index(df, discrimination_vars):
    """
    Compute perceived everyday discrimination index per wave.
    - Reverse-codes all items: 7 - x (6-point scale)
    - Averages across 6 items
    - Sets score to NaN if more than 3 items are missing
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, items in discrimination_vars.items():
        used_vars.extend(items)

        data = df[items].copy()
        data = data.apply(lambda x: 7 - x)  # Reverse-code all items (1–6 → 6–1)

        missing_count = data.isna().sum(axis=1)
        score = data.mean(axis=1)
        score[missing_count > 3] = pd.NA

        result[f"{wave}_discrimination"] = score

    return result, used_vars



def aggregate_vars(df, var_map):
    """
    Creates new variables by taking the first non-missing value from a list of source columns.
    
    Args:
        df (pd.DataFrame): The original dataframe.
        var_map (dict): {new_var_name: [source_var1, source_var2, ...]}
    
    Returns:
        (pd.DataFrame, list): closed variables and list of used source columns.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, sources in var_map.items():
        result[new_var] = df[sources].bfill(axis=1).iloc[:, 0]
        used_vars.extend(sources)

    return result, used_vars


def compute_stressful_events(df, stressful_events):
    """
    close binary stressful event indicators across waves.
    - Returns 1 (yes) if any value is 1
    - Returns 0 (no) if all non-missing values are 0
    - Returns NaN if all values are missing
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in stressful_events.items():
        used_vars.extend(var_list)
        data = df[var_list]

        # Check if any value is 1
        any_yes = data.eq(1).any(axis=1)

        # Check if all values are missing
        all_missing = data.isna().all(axis=1)

        # Set value to 1 if any is 1, else 5
        result[new_var] = 5
        result.loc[any_yes, new_var] = 1

        # Set to NaN if all are missing
        result.loc[all_missing, new_var] = pd.NA

    return result, used_vars

def aggregate_first_nonmissing(df, var_dict):
    """
    Aggregates multiple variables by selecting the first non-missing value across waves (P → K).

    Args:
        df (pd.DataFrame): Your main dataset.
        var_dict (dict): Dictionary of {new_var_name: [list of raw vars in priority order]}.

    Returns:
        pd.DataFrame: DataFrame with new variables.
        list: List of used raw variable names.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in var_dict.items():
        data = df[var_list]
        result[new_var] = data.bfill(axis=1).iloc[:, 0]
        used_vars.extend(var_list)

    return result, used_vars


def compute_work_environment(df, work_env_vars):
    """
    Work environment per wave.
    - Keep 5s (does-not-apply) as-is
    - Reverse-code first item (1–4 only): x -> 5 - x
    - Score = mean of 1–4 items, ignoring 5s
    - Missingness rule: allow up to 2 missing among non-5 cells
    - If no valid (1–4) items: score = 5 if any 5 present; else <NA>
    Returns (result_df, used_vars)
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    def _mean_excluding_5(data, items, allow_missing):
        is_5 = data.eq(5)
        is_valid = data.isin([1, 2, 3, 4])

        # Missing among non-5 cells
        missing_non5 = (data.isna() & (~is_5)).sum(axis=1)

        # Keep only valid 1–4 for the mean
        scorable = data.where(is_valid, pd.NA)
        score = scorable.mean(axis=1)

        # Apply missingness tolerance
        score[missing_non5 > allow_missing] = pd.NA

        # Handle rows with zero valid items
        valid_count = is_valid.sum(axis=1)
        no_valid = valid_count.eq(0)
        only_fives = is_5.any(axis=1) & no_valid
        score[only_fives] = 5
        score[no_valid & ~only_fives] = pd.NA

        return score.astype("Float64")

    for wave, items in work_env_vars.items():
        used_vars.extend(items)

        # reverse-code first item on 1–4 only, leave 5 unchanged
        first = items[0]
        x = df[first]
        rev = x.where(~x.isin([1, 2, 3, 4]), 5 - x)

        tmp = df[items].copy()
        tmp[first] = rev

        score = _mean_excluding_5(tmp, items, allow_missing=2)
        result[f"{wave}_WORK_ENV"] = score

    return result, used_vars


def compute_coworker_support(df, coworker_support_vars):
    """
    Coworker support per wave.
    - Keep 5s
    - Score = mean of 1–4, ignoring 5s
    - Missingness rule: allow 0 missing among non-5 cells
    - If no valid (1–4) items: score = 5 if any 5 present; else <NA>
    Returns (result_df, used_vars)
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    def _mean_excluding_5(data, items, allow_missing):
        is_5 = data.eq(5)
        is_valid = data.isin([1, 2, 3, 4])

        missing_non5 = (data.isna() & (~is_5)).sum(axis=1)
        scorable = data.where(is_valid, pd.NA)
        score = scorable.mean(axis=1)
        score[missing_non5 > allow_missing] = pd.NA

        valid_count = is_valid.sum(axis=1)
        no_valid = valid_count.eq(0)
        only_fives = is_5.any(axis=1) & no_valid
        score[only_fives] = 5
        score[no_valid & ~only_fives] = pd.NA

        return score.astype("Float64")

    for wave, items in coworker_support_vars.items():
        used_vars.extend(items)
        score = _mean_excluding_5(df[items], items, allow_missing=1)
        result[f"{wave}_COWORKER_SUPPORT"] = score

    return result, used_vars


def compute_supervisor_support(df, supervisor_support_vars):
    """
    Supervisor support per wave.
    - Keep 5s
    - Score = mean of 1–4, ignoring 5s
    - Missingness rule: allow up to 1 missing among non-5 cells
    - If no valid (1–4) items: score = 5 if any 5 present; else <NA>
    Returns (result_df, used_vars)
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    def _mean_excluding_5(data, items, allow_missing):
        is_5 = data.eq(5)
        is_valid = data.isin([1, 2, 3, 4])

        missing_non5 = (data.isna() & (~is_5)).sum(axis=1)
        scorable = data.where(is_valid, pd.NA)
        score = scorable.mean(axis=1)
        score[missing_non5 > allow_missing] = pd.NA

        valid_count = is_valid.sum(axis=1)
        no_valid = valid_count.eq(0)
        only_fives = is_5.any(axis=1) & no_valid
        score[only_fives] = 5
        score[no_valid & ~only_fives] = pd.NA

        return score.astype("Float64")

    for wave, items in supervisor_support_vars.items():
        used_vars.extend(items)
        score = _mean_excluding_5(df[items], items, allow_missing=2)
        result[f"{wave}_SUPERVISOR_SUPPORT"] = score

    return result, used_vars


def compute_job_scales(df, job_vars):
    """
    Compute job satisfaction and job stress scale scores for each wave.
    - Keep 5s (does-not-apply) as-is
    - Exclude 5s from the mean (use only 1–4 responses)
    - Apply missingness rules on non-5 items:
        • JOB_SAT: set <NA> if >2 missing among non-5
        • JOB_STRESS: set <NA> if >2 missing among non-5
    - If no valid (1–4) items: score = 5 if any 5 present; else <NA>
    """

    result = pd.DataFrame(index=df.index)
    used_vars = []

    def _mean_excluding_5(data, items, allow_missing):
        is_5 = data.eq(5)
        is_valid = data.isin([1, 2, 3, 4])

        # Missing among non-5 cells
        missing_non5 = (data.isna() & (~is_5)).sum(axis=1)

        # Mean over valid (1–4)
        score = data.where(is_valid, pd.NA).mean(axis=1)
        score[missing_non5 > allow_missing] = pd.NA

        # Handle rows with zero valid items
        valid_count = is_valid.sum(axis=1)
        no_valid = valid_count.eq(0)
        only_fives = is_5.any(axis=1) & no_valid
        score[only_fives] = 5
        score[no_valid & ~only_fives] = pd.NA

        return score.astype("Float64")

    for wave, domains in job_vars.items():
        # --- Satisfaction scale ---
        sat_all = domains['SAT']['all']
        sat_reverse = domains['SAT']['reverse']
        sat_data = df[sat_all].copy()
        used_vars.extend(sat_all)

        # Reverse-code satisfaction items (1–4 only)
        rev_cols = [c for c in sat_reverse if c in sat_data.columns]
        if rev_cols:
            sub = sat_data[rev_cols]
            # Reverse 1–4; leave 5 and others untouched
            sat_data[rev_cols] = sub.where(~sub.isin([1, 2, 3, 4]), 5 - sub)

        sat_score = _mean_excluding_5(sat_data, sat_all, allow_missing=2)
        result[f"{wave}_JOB_SAT"] = sat_score

        # --- Stress scale ---
        stress_all = domains['STRESS']['all']
        stress_data = df[stress_all].copy()
        used_vars.extend(stress_all)

        stress_score = _mean_excluding_5(stress_data, stress_all, allow_missing=2)
        result[f"{wave}_JOB_STRESS"] = stress_score

    return result, used_vars


#### job vars old functions

def compute_job_scales2(df, job_vars):
    """
    Compute job satisfaction and job stress scale scores for each wave.

    Args:
        df (pd.DataFrame): The input dataset.
        job_vars (dict): A dictionary specifying 'satisfaction' and 'stress' items per wave, including reverse-coded ones.

    Returns:
        pd.DataFrame: DataFrame with new scale scores.
        list: List of raw variable names used.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, domains in job_vars.items():
        # --- Satisfaction scale ---
        sat_all = domains['SAT']['all']
        sat_reverse = domains['SAT']['reverse']
        sat_data = df[sat_all].copy()

        # Reverse-code satisfaction items (1–4 scale)
        sat_data[sat_reverse] = sat_data[sat_reverse].apply(lambda x: 5 - x)

        sat_missing = sat_data.isna().sum(axis=1)
        sat_score = sat_data.mean(axis=1)
        sat_score[sat_missing > 2] = pd.NA
        result[f"{wave}_JOB_SAT"] = sat_score

        # --- Stress scale ---
        stress_all = domains['STRESS']['all']
        stress_data = df[stress_all].copy()

        stress_missing = stress_data.isna().sum(axis=1)
        stress_score = stress_data.mean(axis=1)
        stress_score[stress_missing > 2] = pd.NA
        result[f"{wave}_JOB_STRESS"] = stress_score

        # Track all used variables
        used_vars.extend(sat_all + stress_all)

    return result, used_vars

def compute_work_environment2(df, work_env_vars):
    """
    Compute work environment index per wave:
    - Reverse-code item 'Q84p'
    - Recode 5s as missing
    - Mean of items (1–4), set score to missing if ≥3 items are missing

    Args:
        df (pd.DataFrame): Your dataset.
        work_env_vars (dict): {wave_prefix: [list of items]}

    Returns:
        pd.DataFrame: Wave-specific work environment scores.
        list: List of raw variable names used.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, items in work_env_vars.items():
        data = df[items].copy()
        used_vars.extend(items)

        # Recode all '5' responses as missing
        data = data.where(data != 5, pd.NA)

        # Reverse-code first item (Q84p)
        data[items[0]] = data[items[0]].apply(lambda x: 5 - x if pd.notna(x) else pd.NA)

        # Compute score: mean, allow up to 2 missing
        missing_count = data.isna().sum(axis=1)
        score = data.mean(axis=1)
        score[missing_count >= 3] = pd.NA

        result[f"{wave}_WORK_ENV"] = score

    return result, used_vars

def compute_coworker_support2(df, coworker_support_vars):
    """
    Compute coworker support scores per wave.

    Rules:
    - Recode all '5' responses as missing
    - Compute average of items (1–4 scale)
    - Set final score to missing if any item is missing

    Args:
        df (pd.DataFrame): The dataset.
        coworker_support_vars (dict): {wave: list of items}

    Returns:
        pd.DataFrame: New coworker support scores per wave.
        list: List of raw variables used.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, items in coworker_support_vars.items():
        data = df[items].copy()
        used_vars.extend(items)

        # Recode all 5s as missing
        data = data.where(data != 5, pd.NA)

        # Set score to missing if any item is missing
        missing = data.isna().sum(axis=1)
        score = data.mean(axis=1)
        score[missing > 0] = pd.NA

        result[f"{wave}_COWORKER_SUPPORT"] = score

    return result, used_vars

def compute_supervisor_support_old(df, supervisor_support_vars):
    """
    Compute supervisor support scores per wave.

    Rules:
    - Recode all '5' responses as missing
    - Compute average of items (1–4 scale)
    - Set final score to missing if ≥2 items are missing

    Args:
        df (pd.DataFrame): The dataset.
        supervisor_support_vars (dict): {wave: list of items}

    Returns:
        pd.DataFrame: Supervisor support scores per wave.
        list: List of raw variables used.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for wave, items in supervisor_support_vars.items():
        data = df[items].copy()
        used_vars.extend(items)

        # Recode 5s as missing
        data = data.where(data != 5, pd.NA)

        # Set score to missing if ≥2 items are missing
        missing = data.isna().sum(axis=1)
        score = data.mean(axis=1)
        score[missing >= 2] = pd.NA

        result[f"{wave}_SUPERVISOR_SUPPORT"] = score

    return result, used_vars

def convert_weight_height_bmi(df, weight_var='WEIGHT', max_weight_var = 'MAX_WEIGHT', feet_var='HEIGHT_FEET', inch_var='HEIGHT_INCHES'):
    """
    Converts weight and height into metric units and calculates BMI.

    - Weight: pounds → kilograms
    - Height: feet/inches → centimeters
    - BMI: kg / (m^2)

    Args:
        df (pd.DataFrame): Input DataFrame
        weight_var (str): Column name for weight in pounds
        feet_var (str): Column name for height in feet
        inch_var (str): Column name for height in inches

    Returns:
        pd.DataFrame: New columns: WEIGHT_KG, HEIGHT_CM, BMI
        list: List of raw columns used
    """
    result = pd.DataFrame(index=df.index)
    # Weight in kg
    result['WEIGHT_KG'] = df[weight_var] * 0.453592
    result['WEIGHT_KG'] = result['WEIGHT_KG'].where(df[weight_var].notna(), pd.NA)

    result['MAX_WEIGHT_KG'] = df[max_weight_var] * 0.453592
    result['MAX_WEIGHT_KG'] = result['MAX_WEIGHT_KG'].where(df[max_weight_var].notna(), pd.NA)

    # Height in cm
    height_cm = df[feet_var] * 30.48 + df[inch_var] * 2.54
    height_cm[df[[feet_var, inch_var]].isna().any(axis=1)] = pd.NA
    result['HEIGHT_CM'] = height_cm

    # BMI = kg / (m^2)
    height_m = height_cm / 100
    result['BMI'] = result['WEIGHT_KG'] / (height_m ** 2)
    result['BMI'] = result['BMI'].where(height_cm.notna() & df[weight_var].notna(), pd.NA)

    return result


#for counting the number of schools and universities that the particpant attended
def count_nonmissing_if_any(df, num_educ_institutions_vars):
    """
    Count non-missing entries for each variable list in the dictionary.
    If all are missing → NaN.

    Args:
        df (pd.DataFrame): Input dataset
        num_educ_institutions_vars (dict): {'NEW_VAR_NAME': [list of raw vars]}

    Returns:
        pd.DataFrame: New columns with counts
        list: All used raw variables
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in num_educ_institutions_vars.items():
        data = df[var_list]
        count = data.notna().sum(axis=1)
        count[count == 0] = pd.NA  # Replace 0 with NA
        result[new_var] = count
        used_vars.extend(var_list)

    return result, used_vars

def compute_private_education(df, private_vars_dict):
    """
    Create binary indicator for private school/university attendance.

    - 1 = Yes (if any value is 1 in the raw variables)
    - 0 = No (if all values are present and none are 1)
    - NA = Missing (if all values are missing)

    Args:
        df (pd.DataFrame): Dataset.
        private_vars_dict (dict): {'NEW_VAR_NAME': [list of raw vars]}.

    Returns:
        pd.DataFrame: Binary columns (1 = Yes, 0 = No).
        list: Used raw variables.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in private_vars_dict.items():
        data = df[var_list]
        used_vars.extend(var_list)

        any_yes = data.eq(1).any(axis=1)
        all_missing = data.isna().all(axis=1)

        result[new_var] = 0  # Default = No
        result.loc[any_yes, new_var] = 1  # Yes if any 1
        result.loc[all_missing, new_var] = pd.NA  # NA if all missing

    return result, used_vars



def compute_highest_degree(df, highest_degree_vars):
    """
    Computes the highest degree from multiple university degree codes.

    Degree code ranking is hardcoded based on increasing educational level:
    8 < 6 < 7 < 10 < 11 < 1 < 2 < 3 < 9 < 4 < 5

    Args:
        df (pd.DataFrame): Input dataset
        highest_degree_vars (dict): {'HIGHEST_DEGREE': [list of raw vars]}

    Returns:
        pd.DataFrame: New column 'HIGHEST_DEGREE'
        list: List of raw variables used
    """
    degree_order = {
        8: 1,   # Did not complete
        6: 2,   # High school equivalency
        7: 3,   # Other certification
        10: 4,  # Currently enrolled
        11: 5,  # Continuing education
        1: 6,   # Associate’s
        2: 7,   # Bachelor’s
        3: 8,   # Master’s
        9: 9,   # Specialist beyond master’s
        4: 10,  # Doctorate
        5: 10,  # Professional degree – merged with Doctorate
        }   

    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in highest_degree_vars.items():
        used_vars.extend(var_list)
    
    # Remove 96s before replacing values
        data = df[var_list].copy()
        data = data.where(~data.isin([96]), pd.NA)

    # Replace with ordered ranks
        ranked_df = data.replace(degree_order)

    # Compute highest rank
        highest_rank = ranked_df.max(axis=1)

    # Set to NaN if all are missing
        highest_rank[data.isna().all(axis=1)] = pd.NA

        result[new_var] = highest_rank

    return result, used_vars

def compute_unemployment_reasons(df, not_work_reason_vars):
    """
    Create binary indicators for broad categories of unemployment reasons.
    If all values are missing → NaN.

    Args:
        df (pd.DataFrame): Input DataFrame
        reason_vars (list): List of LH38AM* variables

    Returns:
        pd.DataFrame: One column per reason category
        list: List of raw variables used
    """
    data = df[not_work_reason_vars]

    result = pd.DataFrame(index=df.index)
    result['NOTWORK_REASON_FAMILY'] = data.isin([1]).any(axis=1).astype('Int64')
    result['NOTWORK_REASON_HEALTH'] = data.isin([2, 3]).any(axis=1).astype('Int64')
    result['NOTWORK_REASON_UNEMPLOYED'] = data.isin([4]).any(axis=1).astype('Int64')
    result['NOTWORK_REASON_NOT_INTERESTED'] = data.isin([5]).any(axis=1).astype('Int64')

    # Set all to NA if all original values are missing
    all_missing = data.isna().all(axis=1)
    result.loc[all_missing] = pd.NA

    return result, not_work_reason_vars


def compute_living_at10_binaries(df, cols):
    """
    Build 8 binary flags from LH6M* 'check all that apply' items (age 10).
    If any of the columns in `cols` equals the target code, flag=1; else 0.
    Rows with all-NA across `cols` get <NA>.

    Codes (from codebook):
      1 BIOLOGICAL MOTHER            -> LIVING_AT10_BIOMOTH
      2 BIOLOGICAL FATHER            -> LIVING_AT10_BIOFATH
      3 ADOPT/STEP/FOSTER MOTHER     -> LIVING_AT10_STEPMOTH
      4 ADOPT/STEP/FOSTER FATHER     -> LIVING_AT10_STEPFATH
      5 BIO BRO/SIS                   \
      6 ADOPT/STEP/FOST/HALF BRO/SIS   > LIVING_AT10_ANY_SIBS  (5 or 6)
      7 GRANDPARENT(S)               -> LIVING_AT10_GRANDP
      8 OTHER RELATIVE(S)            -> LIVING_AT10_OTHREL
      9 OTHER NON-RELATIVE(S)        -> LIVING_AT10_NONREL
    """
    out = pd.DataFrame(index=df.index)

    # convenience masks
    row_all_na = df[cols].isna().all(axis=1)

    def flag_eq(code):
        m = df[cols].eq(code).any(axis=1)
        s = m.astype("Int64")
        s[row_all_na] = pd.NA
        return s

    def flag_in(codes):
        m = df[cols].isin(codes).any(axis=1)
        s = m.astype("Int64")
        s[row_all_na] = pd.NA
        return s

    out["LIVING_AT10_BIOMOTH"] = flag_eq(1)
    out["LIVING_AT10_BIOFATH"] = flag_eq(2)
    out["LIVING_AT10_STEPMOTH"] = flag_eq(3)
    out["LIVING_AT10_STEPFATH"] = flag_eq(4)
    out["LIVING_AT10_SIBLINGS"] = flag_in([5, 6])   # combines 5 & 6
    out["LIVING_AT10_GRANDP"]   = flag_eq(7)
    out["LIVING_AT10_OTHREL"]   = flag_eq(8)
    out["LIVING_AT10_NONREL"]   = flag_eq(9)

    return out, cols

def compute_living_at_fulljob_binaries(df, cols):
    """
    Creates binary indicators from LH14M* ('lived with when started first full-time job').
    For each respondent, flag = 1 if ANY of the columns in `cols` equals the code,
    0 if none, and <NA> if all `cols` are missing.

    Codes (codebook):
      1  Spouse/partner                  -> LIVING_ATFULLJOB_PARTNER
      2  Biological children             -> LIVING_ATFULLJOB_BIOCHILD
      3  Adopted/foster/step children    -> LIVING_ATFULLJOB_STEPCHILD
      4  Brother(s)/sister(s)            -> LIVING_ATFULLJOB_SIBS
      5  Parent(s)                        -> LIVING_ATFULLJOB_PARENTS
      6  Grandparent(s)                   -> LIVING_ATFULLJOB_GRANDP
      7  Parent(s)-in-law                 -> LIVING_ATFULLJOB_INLAWS
      8  Other relative(s)                -> LIVING_ATFULLJOB_OTHREL
      9  Other non-relative(s)            -> LIVING_ATFULLJOB_NONREL
     10  Lived alone                      -> LIVING_ATFULLJOB_ALONE
     11  Children (unspecified)           -> LIVING_ATFULLJOB_CHILDREN_UNSPEC
     12  In the military (vol)            -> LIVING_ATFULLJOB_MILITARY
     97  Other                            -> LIVING_ATFULLJOB_OTHER
    """

    out = pd.DataFrame(index=df.index)
    all_na = df[cols].isna().all(axis=1)

    def flag_eq(code):
        s = df[cols].eq(code).any(axis=1).astype("Int64")
        s[all_na] = pd.NA
        return s

    out["LIVING_ATFULLJOB_PARTNER"]          = flag_eq(1)
    out["LIVING_ATFULLJOB_BIOCHILD"]         = flag_eq(2)
    out["LIVING_ATFULLJOB_STEPCHILD"]        = flag_eq(3)
    out["LIVING_ATFULLJOB_SIBS"]             = flag_eq(4)
    out["LIVING_ATFULLJOB_PARENTS"]          = flag_eq(5)
    out["LIVING_ATFULLJOB_GRANDP"]           = flag_eq(6)
    out["LIVING_ATFULLJOB_INLAWS"]           = flag_eq(7)
    out["LIVING_ATFULLJOB_OTHREL"]           = flag_eq(8)
    out["LIVING_ATFULLJOB_NONREL"]           = flag_eq(9)
    out["LIVING_ATFULLJOB_ALONE"]            = flag_eq(10)
    out["LIVING_ATFULLJOB_CHILDREN_UNSPEC"]  = flag_eq(11)
    out["LIVING_ATFULLJOB_MILITARY"]         = flag_eq(12)
    # Return (new_vars_df, used_cols) to match your pipeline contract
    return out, cols


def compute_living_at40_binaries(df, cols):
    """
    Creates binary indicators from LH14M* ('lived with when started first full-time job').
    For each respondent, flag = 1 if ANY of the columns in `cols` equals the code,
    0 if none, and <NA> if all `cols` are missing.

    Codes (codebook):
      1  Spouse/partner                  -> LIVING_AT40_PARTNER
      2  Biological children             -> LIVING_AT40_BIOCHILD
      3  Adopted/foster/step children    -> LIVING_AT40_STEPCHILD
      4  Brother(s)/sister(s)            -> LIVING_AT40_SIBS
      5  Parent(s)                        -> LIVING_AT40_PARENTS
      6  Grandparent(s)                   -> LIVING_AT40_GRANDP
      7  Parent(s)-in-law                 -> LIVING_AT40_INLAWS
      8  Other relative(s)                -> LIVING_AT40_OTHREL
      9  Other non-relative(s)            -> LIVING_AT40_NONREL
     10  Lived alone                      -> LIVING_AT40_ALONE
     11  Children (unspecified)           -> LIVING_AT40_CHILDREN_UNSPEC
     12  In the military (vol)            -> LIVING_AT40_MILITARY
     97  Other                            -> LIVING_AT40_OTHER
    """

    out = pd.DataFrame(index=df.index)
    all_na = df[cols].isna().all(axis=1)

    def flag_eq(code):
        s = df[cols].eq(code).any(axis=1).astype("Int64")
        s[all_na] = pd.NA
        return s

    out["LIVING_AT40_PARTNER"]          = flag_eq(1)
    out["LIVING_AT40_BIOCHILD"]         = flag_eq(2)
    out["LIVING_AT40_STEPCHILD"]        = flag_eq(3)
    out["LIVING_AT40_SIBS"]             = flag_eq(4)
    out["LIVING_AT40_PARENTS"]          = flag_eq(5)
    out["LIVING_AT40_GRANDP"]           = flag_eq(6)
    out["LIVING_AT40_INLAWS"]           = flag_eq(7)
    out["LIVING_AT40_OTHREL"]           = flag_eq(8)
    out["LIVING_AT40_NONREL"]           = flag_eq(9)
    out["LIVING_AT40_ALONE"]            = flag_eq(10)
    out["LIVING_AT40_CHILDREN_UNSPEC"]  = flag_eq(11)
    # Return (new_vars_df, used_cols) to match your pipeline contract
    return out, cols


def compute_relationship_history(df, relationship_vars):
    """
    Compute relationship variables from HRS life history data.
    Filters out any relationship (marriage or cohab) starting in 2016 or later.
    """
    result = pd.DataFrame(index=df.index)

    marriage = relationship_vars['MARRIAGE']
    cohab = relationship_vars['COHABITATION']

    # Mask marriages starting in 2016+
    for i in range(1, 6):
        year_col = f'LH36_{i}C'
        mask = df[year_col] >= 2016
        for suffix in ['B', 'C', 'D', 'E']:
            col = f'LH36_{i}{suffix}'
            if col in df.columns:
                df.loc[mask, col] = pd.NA

    # Mask cohabitations starting in 2016+
    for i in range(1, 6):
        year_col = f'LH37_{i}B'
        mask = df[year_col] >= 2016
        for suffix in ['B', 'C', 'D']:
            col = f'LH37_{i}{suffix}'
            if col in df.columns:
                df.loc[mask, col] = pd.NA

    # EVER_MARRIED
    result['EVER_MARRIED'] = df[marriage['EVER_MARRIED_FLAG']].map({1: 1, 5: 0}).astype('Int64')

    # EVER_LIVED_AS_COUPLE
    result['EVER_LIVED_AS_COUPLE'] = df[cohab['EVER_COHAB_FLAG']].map({1: 1, 5: 0}).astype('Int64')

    # STILL_MARRIED
    end_status_df = df[marriage['END_STATUS']]
    result['STILL_MARRIED'] = end_status_df.eq(1).any(axis=1).astype('Int64')
    result.loc[end_status_df.isna().all(axis=1), 'STILL_MARRIED'] = pd.NA

    # EVER_DIVORCED
    result['EVER_DIVORCED'] = end_status_df.eq(3).any(axis=1).astype('Int64')
    result.loc[end_status_df.isna().all(axis=1), 'EVER_DIVORCED'] = pd.NA

    # EVER_WIDOWED
    result['EVER_WIDOWED'] = end_status_df.eq(2).any(axis=1).astype('Int64')
    result.loc[end_status_df.isna().all(axis=1), 'EVER_WIDOWED'] = pd.NA

    # STILL_COHABITING
    cohab_status_df = df[cohab['END_STATUS']]
    result['STILL_COHABITING'] = cohab_status_df.eq(3).any(axis=1).astype('Int64')
    result.loc[cohab_status_df.isna().all(axis=1), 'STILL_COHABITING'] = pd.NA

    # LONGEST_MARRIAGE_YEARS
    start_df = df[marriage['YEAR_MARRIED']]
    end_df = df[marriage['YEAR_ENDED']]
    duration = end_df.subtract(start_df)
    duration = duration.where(end_df.notna(), 2016 - start_df)
    result['LONGEST_MARRIAGE_YEARS'] = duration.max(axis=1)
    result.loc[start_df.isna().all(axis=1), 'LONGEST_MARRIAGE_YEARS'] = pd.NA

    used_vars = (
        marriage['YEAR_MARRIED'] + marriage['YEAR_ENDED'] + marriage['END_STATUS'] +
        cohab['YEAR_START'] + cohab['YEAR_ENDED'] + cohab['END_STATUS'] +
        [marriage['EVER_MARRIED_FLAG'], cohab['EVER_COHAB_FLAG']]
    )

    return result, used_vars




def compute_rand_relationship_history(df, mstat_vars):
    """
    Computes EVER_MARRIED_RAND, EVER_WIDOWED_RAND, EVER_DIVORCED_RAND from RAND marital status variables.

    Args:
        df (pd.DataFrame): DataFrame with RAND marital status columns.
        mstat_vars (list): List of RAND marital status columns (e.g., R1MSTAT ... R13MSTAT).

    Returns:
        pd.DataFrame: New columns for relationship history.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = mstat_vars

    # EVER_MARRIED_RAND: 1 if ever not 8, 0 if all 8, NA if all missing
    data = df[mstat_vars]
    all_missing = data.isna().all(axis=1)
    all_8 = data.eq(8).all(axis=1)
    any_not_8 = data.notna().any(axis=1) & ~all_8

    result['EVER_MARRIED_RAND'] = pd.NA
    result.loc[all_8, 'EVER_MARRIED_RAND'] = 0
    result.loc[any_not_8, 'EVER_MARRIED_RAND'] = 1

    # EVER_WIDOWED_RAND: 1 if any == 7, 0 if none == 7 and at least one not missing, NA if all missing
    any_7 = data.eq(7).any(axis=1)
    result['EVER_WIDOWED_RAND'] = pd.NA
    result.loc[any_7, 'EVER_WIDOWED_RAND'] = 1
    result.loc[~any_7 & ~all_missing, 'EVER_WIDOWED_RAND'] = 0

    # EVER_DIVORCED_RAND: 1 if any == 5, 0 if none == 5 and at least one not missing, NA if all missing
    any_5 = data.eq(5).any(axis=1)
    result['EVER_DIVORCED_RAND'] = pd.NA
    result.loc[any_5, 'EVER_DIVORCED_RAND'] = 1
    result.loc[~any_5 & ~all_missing, 'EVER_DIVORCED_RAND'] = 0

    return result,  used_vars










def compute_job_history(df, hhid_col='HHID'):
    import pandas as pd

    job_data = []

    for i in range(1, 11):
        job_cols = {
            'START_YEAR': f'LH41_{i}A',
            'END_YEAR': f'LH41_{i}B',
            'DM1': f'LH41_{i}DM1',
            'DM2': f'LH41_{i}DM2',
            'DM3': f'LH41_{i}DM3',
            'DM4': f'LH41_{i}DM4',
            'DM5': f'LH41_{i}DM5',
            'DM6': f'LH41_{i}DM6',
            'INDUSTRY': f'LH41_{i}EM',
            'OCCUPATION': f'LH41_{i}FM'
        }

        available_cols = {k: v for k, v in job_cols.items() if v in df.columns}
        subset_cols = [hhid_col] + list(available_cols.values())
        subset = df[subset_cols].copy()
        subset = subset.rename(columns={v: k for k, v in available_cols.items()})
        subset['JOBID'] = i

        for col in available_cols.keys():
            subset[col] = pd.to_numeric(subset[col], errors='coerce')

        if 'START_YEAR' in subset:
            subset['START_YEAR'] = subset['START_YEAR'].replace(99997, pd.NA)
        if 'END_YEAR' in subset:
            subset['END_YEAR'] = subset['END_YEAR'].replace(99997, pd.NA)

        for col in ['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6', 'INDUSTRY', 'OCCUPATION']:
            if col in subset:
                subset[col] = subset[col].replace(997, pd.NA)

        job_data.append(subset)

    job_history_long = pd.concat(job_data, ignore_index=True)

    # ---- Job duration ----
    job_history_long['END_YEAR_FIXED'] = job_history_long['END_YEAR'].replace(9996, 2016)
    job_history_long['END_YEAR_FIXED'] = pd.to_numeric(job_history_long['END_YEAR_FIXED'], errors='coerce')

    job_history_long['JOB_DURATION'] = job_history_long.apply(
        lambda row: row['END_YEAR_FIXED'] - row['START_YEAR']
        if pd.notna(row['START_YEAR']) and pd.notna(row['END_YEAR_FIXED'])
        else pd.NA,
        axis=1)
    job_history_long['JOB_DURATION'] = job_history_long['JOB_DURATION'].where(job_history_long['JOB_DURATION'] >= 0, pd.NA)

    # ---- Longest job flag ----
    job_history_long['RANK_DURATION'] = job_history_long.groupby('HHID')['JOB_DURATION'].rank(method='first', ascending=False)
    job_history_long['LONGEST_JOB'] = (job_history_long['RANK_DURATION'] == 1)

    # ---- Extract main job data from longest job ----
    longest_jobs = job_history_long[job_history_long['LONGEST_JOB']]
    main_info = longest_jobs[['HHID', 'JOB_DURATION', 'OCCUPATION', 'INDUSTRY']].rename(
        columns={
            'JOB_DURATION': 'LONGEST_JOB_YEARS',
            'OCCUPATION': 'MAIN_OCCUPATION',
            'INDUSTRY': 'MAIN_INDUSTRY'
        })

    # ---- Flatten all DM{i} reasons ----
    dm_cols = ['DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6']
    dm_flat = job_history_long[['HHID'] + dm_cols].melt(id_vars='HHID', value_name='REASON')

    def get_reason_flags(sub_df):
        reasons = sub_df['REASON'].dropna().astype('Int64').tolist()
        if not reasons:
            return pd.Series({'REASON_UNEMPLOYED': pd.NA, 'REASON_HEALTH': pd.NA, 'REASON_RETIRED': pd.NA})
        return pd.Series({
            'REASON_UNEMPLOYED': int(4 in reasons),
            'REASON_HEALTH': int(5 in reasons),
            'REASON_RETIRED': int(6 in reasons)
        })

    reasons_flags = (
        dm_flat.groupby('HHID', group_keys=False)
        .apply(get_reason_flags)
        .reset_index()
    )

    # ---- Final summary ----
    summary = main_info.merge(reasons_flags, on='HHID', how='left').drop(columns='HHID')
    job_vars_used = [f'LH41_{i}{sfx}' for i in range(1, 11) for sfx in ['A', 'B', 'C',  'DM1', 'DM2', 'DM3', 'DM4', 'DM5', 'DM6', 'EM', 'FM',  'ELM', 'FLM']]

    return summary, job_vars_used


def summarize_primary_exit_reason(df, main_reason_col='LH49AM1'):
    """
    Maps LH49AM1 (primary job exit reason) to a summary category.
    Self-employment is grouped into 'Voluntary job change'.
    """
    category_map = {
        9: 1, 11: 1,                    # Involuntary separation
        8: 2,                           # Health/disability
        10: 3,                          # Retired
        6: 4,                           # Family reasons
        7: 5,                           # Education
        13: 6,                          # Job dissatisfaction/conflict
        1: 7, 2: 7, 3: 7, 4: 7, 12: 7, 14: 7, 15: 7,  # Voluntary
    }

    result = df[main_reason_col].map(category_map).astype('Int64')
    return result.to_frame('EXIT_JOB_REASON')

def compute_caregiving_variables(df, unpaid_care_vars):
    """
    Computes:
    - NUM_CAREGIVING_EPISODES: number of non-missing relationship entries (care episodes)

    """
    rel_vars = unpaid_care_vars['relationship_vars']
    result = pd.DataFrame(index=df.index)

        # Count number of caregiving episodes based on relationship variables
    count = df[rel_vars].notna().sum(axis=1)
    count[df[rel_vars].isna().all(axis=1)] = pd.NA
    result['NUM_CAREGIVING_EPISODES'] = count

    return result, rel_vars

def aggregate_first_valid(df, var_dict, invalid_values=None):
    """
    Aggregates multiple variables by selecting the first non-missing value across waves (P → K),
    treating specified values (e.g., 8, 9 or x > 50) as missing.

    Args:
        df (pd.DataFrame): Your main dataset.
        var_dict (dict): Dictionary of {new_var_name: [list of raw vars in priority order]}.
        invalid_values (list or function, optional): Values or condition to treat as missing.

    Returns:
        pd.DataFrame: DataFrame with new variables.
        list: List of used raw variable names.
    """
    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in var_dict.items():
        data = df[var_list].copy()
        used_vars.extend(var_list)

        # Handle invalid values
        if callable(invalid_values):
            data = data.mask(data.applymap(invalid_values))
        elif invalid_values is not None:
            data = data.where(~data.isin(invalid_values), pd.NA)

        result[new_var] = data.bfill(axis=1).iloc[:, 0]

    return result, used_vars

def compute_salary_year(df, amount_col='LH47A', unit_col='LH47B'):
    """
    Converts reported salary to annual salary (SALARY_YEAR) based on time unit,
    accounting for special missing codes.

    Parameters:
    - df: DataFrame with salary and unit
    - amount_col: Column with reported earnings
    - unit_col: Column with unit indicator (1=hour, 2=week, 3=month, 4=year)

    Returns:
    - DataFrame with SALARY_YEAR
    - List of raw variables used
    """
    result = pd.DataFrame(index=df.index)

    # Define valid unit-to-multiplier mappings
    multipliers = {
        1: 2080,  # hourly → annual
        2: 52,    # weekly → annual
        3: 12,    # monthly → annual
        4: 1      # already annual
    }

    # Define special missing codes
    missing_amounts = {9999999, 99999997}
    missing_units = {97, 98, 99}

    # Replace special codes with NaN
    clean_amounts = df[amount_col].replace(missing_amounts, pd.NA)
    clean_units = df[unit_col].replace(missing_units, pd.NA)

    # Map time unit to multiplier
    multiplier_series = clean_units.map(multipliers)

    # Calculate salary
    result['SALARY_YEAR'] = clean_amounts * multiplier_series

    # Ensure rows with missing values are set to NaN
    result['SALARY_YEAR'] = result['SALARY_YEAR'].where(
        clean_amounts.notna() & clean_units.notna(), pd.NA
    )

    return result, [amount_col, unit_col]

def get_ancestry_flag_simple(df, var_a, var_e, var_h):
    """
    Returns ANCESTRY_FLAG for each row:
    1 = African (A5_), 2 = European (E5_), 3 = Hispanic (H5_), pd.NA if all missing.

    Args:
        df (pd.DataFrame): DataFrame with the three ancestry columns.
        var_a (str): Column name for African ancestry (A5_XXX).
        var_e (str): Column name for European ancestry (E5_XXX).
        var_h (str): Column name for Hispanic ancestry (H5_XXX).

    Returns:
        pd.Series: ANCESTRY_FLAG for each row.
    """
    flag = pd.Series(pd.NA, index=df.index)

    flag[df[var_a].notna()] = 1
    flag[df[var_e].notna()] = 2
    flag[df[var_h].notna()] = 3
    return flag

def compute_discrimination_reasons(df, reason_vars):
    """
    Creates 10 binary variables indicating whether a respondent ever experienced discrimination
    for specific reasons across any wave.

    Values:
    - 1 if any of the related variables equals the target code
    - 0 if none equal the code and at least one is not missing
    - NA if all related variables are missing
    """
    result = pd.DataFrame(index=df.index)

    # Mapping of target variable name to reason code
    reason_map = {
        'EVER_ANCESTRY_DISCRIMINATION': 1,
        'EVER_GENDER_DISCRIMINATION': 2,
        'EVER_RACE_DISCRIMINATION': 3,
        'EVER_AGE_DISCRIMINATION': 4,
        'EVER_RELIGION_DISCRIMINATION': 5,
        'EVER_WEIGHT_DISCRIMINATION': 6,
        'EVER_DISABILITY_DISCRIMINATION': 7,
        'EVER_APPAREANCE_DISCRIMINATION': 8,
        'EVER_ORIENTATION_DISCRIMINATION': 9,
        'EVER_FINANCIAL_DISCRIMINATION': 10,
    }


    # Ensure only existing columns are used
    existing_vars = [v for v in reason_vars if v in df.columns]
    data = df[existing_vars]

    for var_name, code in reason_map.items():
        any_match = data.eq(code).any(axis=1)
        all_missing = data.isna().all(axis=1)
        result[var_name] = 0
        result.loc[any_match, var_name] = 1
        result.loc[all_missing, var_name] = pd.NA

    return result, existing_vars


def map_move_reason(code):
    if code in [1, 2, 9, 34]:
        return 1  # Family/Social
    elif code in [3, 33, 41]:
        return 2  # Health/Support
    elif code in [4, 5, 40]:
        return 3  # Lifestyle
    elif code in [6, 7, 23]:
        return 4  # Housing space
    elif code in [8, 11]:
        return 5  # Employment
    elif code in [10, 12, 30]:
        return 6  # Access/Convenience
    elif code in [20, 21, 22, 32]:
        return 7  # Financial
    elif code in [24, 25, 26]:
        return 8  # Discontent
    elif pd.isna(code):
        return pd.NA
    else:
        return pd.NA  # Other



def compute_household_income(df, income_vars, income_threshold=20000):
    """
    Constructs life-course income measures using household total income (HwITOT):
    1. HH_MEAN_INCOME: average household income across waves
    2. HH_PERSISTENT_LOW_INCOME: number of waves with income below a threshold
    3. HH_INCOME_DROPS: number of wave-to-wave income drops of 50% or more

    Args:
        df (pd.DataFrame): Input dataframe
        income_vars (list): List of income variable names
        income_threshold (int): Threshold to define low income (default: $20,000)

    Returns:
        pd.DataFrame: DataFrame with 3 new variables
        list: List of raw income variables used
    """
    df_income = df[income_vars].copy()

    # 1. Mean income
    mean_income = df_income.mean(axis=1, skipna=True)

    # 2. Persistent low income
    persistent_low_income = df_income.lt(income_threshold).sum(axis=1)

    # 3. Sudden income drops
    sudden_drops_list = []
    for i in range(1, len(income_vars)):
        prev = df_income[income_vars[i - 1]]
        curr = df_income[income_vars[i]]
        drop = ((curr < 0.5 * prev) & prev.notna() & curr.notna())
        sudden_drops_list.append(drop)

    sudden_income_drops = pd.concat(sudden_drops_list, axis=1).sum(axis=1)

    result = pd.DataFrame(index=df.index)
    result['HH_MEAN_INCOME'] = mean_income
    result['HH_PERSISTENT_LOW_INCOME'] = persistent_low_income
    result['HH_INCOME_DROPS'] = sudden_income_drops

    return result, income_vars


def compute_unemployment(df, unem_vars, chronic_threshold=2):
    """
    Constructs unemployment-related life-course variables:
    1. NUM_UNEMPLOYMENT: number of waves in which respondent received unemployment income
    2. CHRONIC_UNEMPLOYMENT: indicator for respondents with unemployment income in >= chronic_threshold waves

    Parameters:
        df (pd.DataFrame): Input DataFrame containing RAND unemployment income variables.
        wave_nums (list): List of wave numbers to include (default: 1 to 13).
        prefix (str): Prefix for variable names (default: 'R').
        var_root (str): Root of the unemployment variable (default: 'IUNEM').
        chronic_threshold (int): Number of waves to define "chronic" unemployment.

    Returns:
        pd.DataFrame: Two new variables: NUM_UNEMPLOYMENT, CHRONIC_UNEMPLOYMENT
        list: List of raw variable names used
    """
    df_unem = df[unem_vars].copy()

    # Binary flags: 1 if received unemployment income (>0), else 0
    unem_flags = df_unem.gt(0).astype(int)

    # Number of waves with unemployment income
    num_unemployment = unem_flags.sum(axis=1)

    # Chronic if unemployed in >= chronic_threshold waves
    chronic_unemployment = (num_unemployment >= chronic_threshold).astype('Int64')
    chronic_unemployment[num_unemployment.isna()] = pd.NA

    result = pd.DataFrame(index=df.index)
    result['NUM_UNEMPLOYMENT'] = num_unemployment
    result['CHRONIC_UNEMPLOYMENT'] = chronic_unemployment

    return result, unem_vars



def aggregate_ever_vars_str(df, var_dict, invalid_values=[96, 97, 98, 99]):
    """
    Aggregates binary variables:
    - Returns 1 if any value is 1
    - Returns 0 if no 1s and at least one valid non-missing value
    - Returns NaN if all values are missing (or invalid)

    Parameters:
        df (pd.DataFrame): Input DataFrame
        var_dict (dict): {new_variable_name: [list of source variables]}
        invalid_values (list): Values to convert to NaN before aggregation

    Returns:
        pd.DataFrame: New binary variables
        list: List of raw variables used
    """
    

    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in var_dict.items():
        data = df[var_list].copy()
        data = data[var_list].applymap(lambda x: 1 if str(x).startswith("1") else 0 if str(x).startswith("0") else pd.NA)
        used_vars.extend(var_list)

        # Set specified invalid values to NaN
        data = data.replace({5: 0})
        data = data.mask(data.isin(invalid_values))

        # Aggregation logic
        any_1 = data.eq(1).any(axis=1)
        all_missing = data.isna().all(axis=1)

        result[new_var] = pd.NA  # Start with missing
        result.loc[~all_missing & ~any_1, new_var] = 0  # No 1s, but at least one valid
        result.loc[any_1, new_var] = 1  # At least one 1

    return result, used_vars



def aggregate_ever_vars_bin(df, var_dict, invalid_values=[96, 97, 98, 99]):
    """
    Aggregates binary variables:
    - Returns 1 if any value is 1
    - Returns 0 if no 1s and at least one valid non-missing value
    - Returns NaN if all values are missing (or invalid)

    Parameters:
        df (pd.DataFrame): Input DataFrame
        var_dict (dict): {new_variable_name: [list of source variables]}
        invalid_values (list): Values to convert to NaN before aggregation

    Returns:
        pd.DataFrame: New binary variables
        list: List of raw variables used
    """
    

    result = pd.DataFrame(index=df.index)
    used_vars = []

    for new_var, var_list in var_dict.items():
        data = df[var_list].copy()
        used_vars.extend(var_list)

        # Recode 5 as 0
        data = data.replace({5: 0})

        # Replace invalid values with NaN
        data = data.where(~data.isin(invalid_values), pd.NA)

        # Aggregation logic
        any_one = data.eq(1).any(axis=1)
        all_missing = data.isna().all(axis=1)

        result[new_var] = pd.NA
        result.loc[any_one, new_var] = 1
        result.loc[~any_one & ~all_missing, new_var] = 0

    return result, used_vars

def compute_lh50_score(df):
    """
    Compute composite job satisfaction score at age 30–40 from LH50A–LH50H.
    Reverse-codes items where higher values reflect *worse* job quality.

    Parameters:
    df : pandas.DataFrame
        Must contain LH50A to LH50H.

    Returns:
    result : pandas.DataFrame
        A DataFrame with the computed JOB_SAT_30 score.
    required_cols : list of str
        The columns used in the computation.
    """
    required_cols = ['LH50A', 'LH50B', 'LH50C', 'LH50D', 'LH50E', 'LH50F', 'LH50G', 'LH50H']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Copy only required columns to avoid side effects
    subset = df[required_cols].copy()
    subset = subset.replace(5, pd.NA)

    # Reverse-code negatively valenced items
    reverse_cols = ['LH50A', 'LH50B', 'LH50G']  # These should be reversed per your version
    for col in reverse_cols:
        subset[col] = 5 - subset[col]

    # Compute average score
    result = pd.DataFrame(index=df.index)
    result['JOB_SAT_30'] = subset.mean(axis=1)

    return result, required_cols

def combine_ethnicities(df, pgs_vars, prefix='PGI_', standardize=False):
    """
    Combina las columnas A5_, E5_ y H5_ en una sola por trait, sin modificar el DataFrame original.

    Args:
        df (pd.DataFrame): DataFrame original con columnas de PGS separadas por etnia.
        pgs_dict (dict): Diccionario con claves como 'AFBC_SOCGEN16' y valores como listas de columnas A5_, E5_, H5_.
        prefix (str): Prefijo para las columnas de salida.

    Returns:
        scores (pd.DataFrame): DataFrame con una columna combinada por trait.
        used_vars (dict): Diccionario con los nombres de columnas usadas para cada trait.
    """
    scores = pd.DataFrame(index=df.index)
    used_vars = []

    for base_name, cols in pgs_vars.items():
        valid_cols = [col for col in cols if col in df.columns]
        if valid_cols:
            combined = df[valid_cols].sum(axis=1, skipna=True)
            all_na_mask = df[valid_cols].isna().all(axis=1)
            combined[all_na_mask] = pd.NA
            scores[f'{prefix}{base_name}'] = combined

            for col in valid_cols:
                if col not in used_vars:
                    used_vars.append(col)
        # Standardize if requested
    if standardize and not scores.empty:
        scaler = StandardScaler()
        scores[:] = scaler.fit_transform(scores)

    return scores, used_vars


def compute_siblings(df, siblings_vars):
    result = pd.DataFrame(index=df.index)
    used_vars = []
    for new_var, var_list in siblings_vars.items():
        # Only use columns that exist in df
        valid_cols = [col for col in var_list if col in df.columns]
        sib_bin = df[[col for col in sibling_binary_cols if col in valid_cols]]
        sib_count = df[[col for col in sibling_count_cols if col in valid_cols]]

        any_6 = sib_bin.eq(6).any(axis=1)
        any_1_2 = sib_bin.isin([1, 2]).any(axis=1)
        any_count = sib_count.gt(0).any(axis=1)

        out = pd.Series(pd.NA, index=df.index)
        out[any_6] = 0
        out[any_1_2 | any_count] = 1
        all_missing = sib_bin.isna().all(axis=1) & sib_count.isna().all(axis=1)
        out[all_missing] = pd.NA

        result[new_var] = out
        used_vars.extend(valid_cols)
    return result, used_vars



def convert_to_numeric(df, columns):
    """
    Convierte columnas tipo string con formato 'X.algo' o 'X algo' a float X.
    Si ya es número, lo deja. Si no se puede convertir, devuelve NaN.
    """
    import re
    import numpy as np
    def clean_numeric(val):
        # Si ya es número, devuélvelo
        if isinstance(val, (int, float)):
            return val
        if pd.isnull(val):
            return np.nan
        try:
            val_str = str(val).strip()
            # Caso exacto tipo '2.0' o '3'
            if re.fullmatch(r'\d+(\.\d+)?', val_str):
                return float(val_str)
            # Extraer número al inicio de string tipo '2.0 drinks per day'
            match = re.match(r'^(\d+(\.\d+)?)', val_str)
            if match:
                return float(match.group(1))
        except:
            pass
        return np.nan

    # Aplicar a cada columna
    for col in columns:
        df[col] = df[col].apply(clean_numeric)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df





##########################################################################################################
# === 3. Running the functions ===
##########################################################################################################




def main():
    # The notebook used an in-memory master_df. Reading CSV may infer dtypes differently.
    # PARITY TEST REQUIRED: compare dtypes and resulting events_df with the notebook run.
    master_df = pd.read_csv(COMBINED_EVENTS_FILE)

    master_df = convert_to_numeric(master_df, rand_vars_str)
    network_scores, network_raws = aggregate_first_nonmissing(master_df, network_vars)

    master_df['ANCESTRY_FLAG'] = get_ancestry_flag_simple(master_df, 'A5_AFBC_SOCGEN16', 'E5_AFBC_SOCGEN16', 'H5_AFBC_SOCGEN16')

    close_ties_scores, close_ties_raws = aggregate_first_valid(
        master_df,
        close_ties_vars,
        invalid_values=lambda x: x > 50
    )

    contact_scores, contact_raws = compute_contact_frequency(master_df, contact_vars)

    support_scores, support_raws = compute_perceived_social_support(master_df, support_vars)

    loneliness_scores, loneliness_raws = compute_loneliness_index(master_df, loneliness_vars)

    neigh_scores, neigh_raws = compute_neighborhood_scales(master_df, neighborhood_vars)

    discrimination_scores, discrimination_raws = compute_discrimination_index(master_df, discrimination_vars)

    reasons_discrimination_scores, reasons_discrimination_raws = compute_discrimination_reasons(master_df, reasons_discrimination_vars)

    unusual_living_scores, unusual_living_raws = aggregate_first_nonmissing(master_df, unusual_living_vars)

    stressful_events_scores, stressful_events_raws = compute_stressful_events(master_df, stressful_events_vars)

    ongoing_stressors_scores, ongoing_stressors_raws = aggregate_first_nonmissing(master_df, ongoing_stressors_vars)

    job_stressors_scores, job_stressors_raws = compute_job_scales(master_df, job_stressors_vars)

    job_satisfied_scores, job_satisfied_raws = aggregate_first_valid(master_df, job_satisfied_vars, invalid_values=[5])

    work_env_scores, work_env_raws = compute_work_environment(master_df, work_env_vars)

    coworker_support_scores, coworker_support_raws = compute_coworker_support(master_df, coworker_support_vars)

    supervisor_support_scores, supervisor_support_raws = compute_supervisor_support(master_df, supervisor_support_vars)

    health_behavior_scores, health_behavior_raws = aggregate_first_valid(master_df, health_behavior_vars, invalid_values = [999, 98, 99, 9998])

    antropometric_scores_1, antropometric_raws_1 = aggregate_first_valid(master_df, antropometric_vars_1, invalid_values= [998, 999, 9998, 9999, 99998, 99999, 999998, 999999, 9999998, 9999999, 99999998, 99999999, 9999999998, 9999999999, 9999999998.0, 9999999999.0])

    antropometric_scores_2, antropometric_raws_2 = aggregate_first_valid(master_df, antropometric_vars_2, invalid_values = [8, 9])

    antropometric_scores_3, antropometric_raws_3 = aggregate_first_valid(master_df, antropometric_vars_3, invalid_values = [98, 99, 99998])


    num_educ_institutions_scores, num_educ_institutions_raws = count_nonmissing_if_any(master_df, num_educ_institutions_vars)

    private_educ_scores, private_educ_raws = compute_private_education(master_df, private_educ_vars)

    highest_degree_scores, highest_degree_raws = compute_highest_degree(master_df, degree_vars)
 
    not_work_reason_scores, not_work_reason_raws = compute_unemployment_reasons(master_df, not_work_reason_vars)

    living_at10_var_scores, living_at10_var_raws = compute_living_at10_binaries(master_df, living_at10_vars)

    living_at20_var_scores, living_at20_var_raws = compute_living_at_fulljob_binaries(master_df, lived_at20_vars)

    living_at40_var_scores, living_at40_var_raws = compute_living_at40_binaries(master_df, lived_at40_vars)

    relationship_history_scores, relationship_history_raws = compute_relationship_history(master_df, relationship_vars) 

    job_history_scores, job_history_raws = compute_job_history(master_df)

    exit_job_reason_scores = summarize_primary_exit_reason(master_df, main_reason_col='LH49AM1')

    unpaid_care_scores, unpaid_care_raws = compute_caregiving_variables(master_df, unpaid_care_vars)

    aggregate_first_valid_scores, aggregate_first_valid_raws = aggregate_first_valid(master_df, aggregate_first_valid_vars, invalid_values=[7, 8, 9])


    salary_year_scores, salary_year_raws = compute_salary_year(master_df, amount_col='LH47A', unit_col='LH47B')

    income_scores, income_raws = compute_household_income(master_df, hh_income_vars, income_threshold=20000)

    unemployment_scores, unemployment_raws = compute_unemployment(master_df, unem_vars)

    ever_bin_scores, ever_bin_raws = aggregate_ever_vars_bin(master_df, ever_vars_bin)

    job_sat_30_scores, job_sat_30_raws = compute_lh50_score(master_df)

    pgs_scores, pgs_raws = combine_ethnicities(master_df, pgs_vars, prefix='PGI_', standardize=True)

    siblings_scores, siblings_raws = compute_siblings(master_df, siblings_vars)

    rand_scores, rand_raws = aggregate_first_nonmissing(master_df, rand_vars)




    #Combine all new scales into one DataFrame
    new_scales = pd.concat([
        network_scores, 
        close_ties_scores, 
        contact_scores, 
        support_scores, 
        loneliness_scores, 
        neigh_scores, 
        discrimination_scores, 
        reasons_discrimination_scores,
        unusual_living_scores, 
        stressful_events_scores, 
        ongoing_stressors_scores,
        job_stressors_scores, 
        job_satisfied_scores,
        work_env_scores, 
        coworker_support_scores, 
        supervisor_support_scores, 
        health_behavior_scores, 
        antropometric_scores_1,
        antropometric_scores_2,
        antropometric_scores_3, 
        num_educ_institutions_scores, 
        private_educ_scores, 
        highest_degree_scores, 
        not_work_reason_scores,  
        living_at10_var_scores, 
        living_at20_var_scores, 
        living_at40_var_scores,
        relationship_history_scores, 
        job_history_scores, 
        exit_job_reason_scores, 
        unpaid_care_scores,  
        aggregate_first_valid_scores, 
        salary_year_scores, 
        unemployment_scores, 
        income_scores, 
        ever_bin_scores,
        job_sat_30_scores,  
        pgs_scores, 
        rand_scores, 
        siblings_scores, 
        ], axis=1)

    #Combine all raw variables to drop
    raw_vars_used = set(network_raws + 
                        close_ties_raws + 
                        contact_raws + 
                        support_raws + 
                        loneliness_raws + 
                        neigh_raws + 
                        discrimination_raws + 
                        reasons_discrimination_raws +
                        unusual_living_raws +
                        stressful_events_raws + 
                        ongoing_stressors_raws +
                        ongoing_stressors_raws +
                        job_stressors_raws +
                        job_satisfied_raws +
                        work_env_raws +
                        coworker_support_raws +
                        supervisor_support_raws +
                        health_behavior_raws +
                        antropometric_raws_1 + 
                        antropometric_raws_2 + 
                        antropometric_raws_3 + 
                        num_educ_institutions_raws +
                        private_educ_raws +
                        highest_degree_raws +
                        not_work_reason_raws +
                        living_at10_var_raws +
                        living_at20_var_raws +
                        living_at40_var_raws +
                        relationship_history_raws +
                        job_history_raws +
                        unpaid_care_raws +
                        aggregate_first_valid_raws +
                        unemployment_raws +
                        income_raws +
                        salary_year_raws +
                        ever_bin_raws +
                        job_sat_30_raws +
                        pgs_raws +
                        rand_raws + 
                        siblings_raws +
                        list(to_delete_raw))



    ##########################################################################################################
    # === 4.  Create events_df with all the new scores ===
    ##########################################################################################################


    events_df = master_df.drop(columns=raw_vars_used).copy()
    events_df = pd.concat([events_df, new_scales], axis=1)





    ##########################################################################################################
    # === 5.  Last modifications ===
    ##########################################################################################################

    scores_to_aggregate, scores_to_aggregate_raws = aggregate_first_nonmissing(events_df, scores_to_aggregate_vars)
    bmi_metrics_scores = convert_weight_height_bmi(events_df)

    events_df = pd.concat([events_df, scores_to_aggregate, bmi_metrics_scores], axis=1)

    events_df['REASON_MOVED_CATEGORY'] = events_df['REASON_MOVED'].apply(map_move_reason)
    events_df[['FAMFIN', 'FAUNEM','MOWORK', 'RTHLTHCH', 'USBORN', 'PACKS_NOW', 'ALC_CUTDOWN', 'ALC_CRITICIZED', 'ALC_GUILT', 'ALC_HANGOVER_DRINK']] = events_df[['FAMFIN', 'FAUNEM', 'MOWORK', 'RTHLTHCH', 'USBORN', 'PACKS_NOW', 'ALC_CUTDOWN', 'ALC_CRITICIZED', 'ALC_GUILT', 'ALC_HANGOVER_DRINK']].replace([8, 9], pd.NA)
    events_df[['FAEDUC', 'MOEDUC', 'NUM_CHILDREN_ALIVE', 'SCHLYRS', 'AGE_STARTED_SMOKING_YRS', 'MAX_PACKS_PER_DAY', 'YEARS_SINCE_STOP_SMOKING', 'AGE_STOPPED_SMOKING', 'DRINKS_PER_DAY', 'LH3A', 'LH9', 'LH13', 'LH16', 'LH19', 'LH33A', 'LH38', 'LH43', 'LH48', 'LH49', 'LH51', 'LH15']] = events_df[['FAEDUC', 'MOEDUC', 'NUM_CHILDREN_ALIVE', 'SCHLYRS', 'AGE_STARTED_SMOKING_YRS', 'MAX_PACKS_PER_DAY', 'YEARS_SINCE_STOP_SMOKING', 'AGE_STOPPED_SMOKING', 'DRINKS_PER_DAY', 'LH3A', 'LH9', 'LH13', 'LH16', 'LH19', 'LH33A', 'LH38', 'LH43', 'LH48', 'LH49', 'LH51', 'LH15']].replace([96, 97, 98, 99], pd.NA)

    events_df[['LH39A', 'LH39B', 'LH39C', 'LH39D', 'LH40A', 'LH40B', 'LH40C', 'LH40D']] = events_df[['LH39A', 'LH39B', 'LH39C', 'LH39D', 'LH40A', 'LH40B', 'LH40C', 'LH40D']].replace([97], pd.NA)

    events_df[['CIGS_NOW', 'MAX_CIGS_PER_DAY', 'LH7', 'LH45M', 'LH60M1', 'LH60M2', 'LH60M3', 'LH42', 'LH46M']] = events_df[['CIGS_NOW', 'MAX_CIGS_PER_DAY', 'LH7', 'LH45M', 'LH60M1', 'LH60M2', 'LH60M3', 'LH42', 'LH46M']].replace([996, 997, 998, 999], pd.NA)
    events_df[['BINGE_DAYS_3M']] = events_df[['BINGE_DAYS_3M']].replace(92, pd.NA)
    events_df[['AGE_STARTED_SMOKING_YEARS', 'YEAR_STOPPED_SMOKING']] = events_df[['AGE_STARTED_SMOKING_YEARS', 'YEAR_STOPPED_SMOKING']].replace(9998, pd.NA)

    events_df['AGE_MAX_WEIGHT'] = events_df['AGE_MAX_WEIGHT'].replace(9999999996, pd.NA)


    events_df['MAX_WEIGHT_KG'] = events_df['MAX_WEIGHT_KG'].mask(events_df['MAX_WEIGHT_KG'] > 300, pd.NA)

    events_df['CLOSE_CHILDREN'] = events_df['CLOSE_CHILDREN'].mask(events_df['CLOSE_CHILDREN'] > 30 , pd.NA)
    events_df['CLOSE_FRIENDS'] = events_df['CLOSE_FRIENDS'].mask(events_df['CLOSE_FRIENDS'] < 0 , pd.NA)
    events_df['RAEVBRN_CAT'] = events_df['RAEVBRN'].apply(lambda x: x if x < 3 else 3)

    
    #recoding binary variables
    binary_vars = ['RELATIVES_NEIGHBORHOOD', 'HOMELESS_AGG', 'JAIL_AGG', 'DISMJOB', 'NHIREDJOB', 'DENPROM', 'PREVMOV', 'DENLOAN', 'UNFPOLICE', 'DENCARE', 'CHLDIED', 'DISASTER', 'COMBAT', 'DRUGOTH', 'ATTACK', 'ILLSELF', 'ILLOTH', 'SCHLOVER', 'TRPOLICE', 'DRKDRUG', 'PHYABUSE', 'LOST_JOB_5Y_AGG', 'UNEMPLOYED_5Y_AGG', 'HH_UNEMPLOYED_5Y_AGG', 'MOVED_5Y_AGG', 'ROBBED_5Y_AGG', 'FRAUD_5Y_AGG', 'CURRENTLY_WORKING', 'JOB_LOCK_MONEY', 'JOB_LOCK_INSURANCE', 'MOVFIN', 'FMFINH', 'LIVEGPAR', 'CHSMOKE', 'CHMISSCH', 'PARSMOKE', 'LH1', 'LH1B1', 'LH1B2', 'LH1B3', 'LH1B4','LH2A', 'LH2B', 'LH2C', 'LH2D', 'LH2E', 'LH2F', 'LH2G', 'LH2H', 'LH2I','LH4A', 'LH4B', 'LH4C', 'LH4D', 'LH4E', 'LH4F', 'LH4G', 'LH23', 'LH30', 'LH31A', 'LH31B', 'LH31C', 'LH31D', 'LH31E', 'LH31F', 'LH31G', 'LH31H', 'LH31I', 'LH31J', 'LH48', 'LH49', 'LH51', 'CURRENTLY_SMOKING', 'ALC_CUTDOWN', 'CURRENTLY_DRINKS', 'ALC_CRITICIZED', 'ALC_GUILT', 'ALC_HANGOVER_DRINK', 'MARRIED_SINCE_LAST', 'DIVORCED_WIDOWED_SINCE_LAST', 'LH38', 'LH34', 'LH39A', 'LH39B', 'LH39C', 'LH39D', 'LH40A', 'LH40B', 'LH40C', 'LH40D', 'HAS_CHILDREN', 'HAS_FAMILY', 'HAS_FRIENDS', 'LIVES_WITH_PARTNER'
    ]




    def recode_binary(df, columns):
        """
        Recode binary variables where 1 = Yes and 5 = No into 1 = Yes, 0 = No.
        Any other values are set to NaN. Missing values are left unchanged .

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (list): List of column names to recode.

        Returns:
            pd.DataFrame: Updated DataFrame with recoded columns.
        """
        df = df.copy()
        for col in columns:
            df[col] = df[col].where(df[col].isin([1, 5, 6]), pd.NA)  # set non-1/5 to NA
            df[col] = df[col].replace({5: 0})  # recode 5 to 0
        return df




    ##########################################################################################################
    # === 6.  Create the final version of events_df  and save it ===
    ##########################################################################################################

    events_df = recode_binary(events_df, binary_vars)



    #JAIL_AGG AND LH4A MERGED!
    events_df.loc[events_df['LH4A'] == 1, 'JAIL_AGG'] = 1

    ### DEALING WITH MISSING VALUES
    events_df = events_df.drop(columns=['AGEDRUGE', 'AGEDRUGD', 'KNOWDNDECEASEDSOURCE', 'EXDEATHMO', 'EXDEATHYR', 'KNOWNDECEASEMO', 'LH60M3', 'LH60M2', 'LH60M1', 'LH42', 'AGE_STARTED_SMOKING_YEARS', 'AGE_STARTED_SMOKING_AGO', 'REASON_UNEMPLOYED', 'REASON_HEALTH', 'REASON_RETIRED', 'LH4A'], errors='ignore')

    def clean_lh1_data(df):
        subvars = ['LH1B1', 'LH1B2', 'LH1B3', 'LH1B4']
        df = df.copy()

        # Paso 1: Si LH1 == 0 → subvars = 0
        df.loc[df['LH1'] == 0, subvars] = df.loc[df['LH1'] == 0, subvars].fillna(0)

        # Paso 2: Si LH1 es NaN, inferirlo desde subvars
        mask_lh1_na = df['LH1'].isna()
        all_zero = df.loc[mask_lh1_na, subvars].eq(0).all(axis=1)
        any_one = df.loc[mask_lh1_na, subvars].eq(1).any(axis=1)

        df.loc[mask_lh1_na & all_zero, 'LH1'] = 0
        df.loc[mask_lh1_na & any_one, 'LH1'] = 1

        # Paso 3: Si LH1 == 1 y alguna subvar == 1 → rellenar NaN con 0
        mask_lh1_1 = df['LH1'] == 1
        has_any_1 = df[subvars].eq(1).any(axis=1)
        df.loc[mask_lh1_1 & has_any_1, subvars] = df.loc[mask_lh1_1 & has_any_1, subvars].fillna(0)

        return df


    def does_not_apply(df, condition_var, condition_value, target_vars, new_value):
        """
        Sets the target variable(s) to a new value where condition_var == condition_value.

        Args:
            df (pd.DataFrame): The DataFrame to modify
            condition_var (str): The variable to check the condition on (e.g., 'LH38')
            condition_value (int or list): The value(s) that trigger the overwrite (e.g., 1 or [1, 2])
            target_vars (list): List of column names to overwrite
            new_value: The value to assign to target_vars when the condition is met

        Returns:
            pd.DataFrame: Modified copy of the original DataFrame
        """
        df_copy = df.copy()
        if not isinstance(condition_value, list):
            condition_value = [condition_value]

        mask = df_copy[condition_var].isin(condition_value)
        df_copy.loc[mask, target_vars] = new_value

        return df_copy

    import pandas as pd



        # Adapt marital status variable
    events_df['MARITAL_STATUS_RAND'] = events_df['MARITAL_STATUS_RAND'].replace({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 4, 8: 5})



    # $$$$$$$$$$$$$ DOES NOT APPLY SECTION $$$$$$$$$

    # ADDING A NEW CATEGORY FOR JOB VARIABLES, "99 MEANS UNKWNOWN"
    for col in ['LONGEST_JOB_INDUSTRY', 'LONGEST_JOB_OCCUPATION']:
        if pd.api.types.is_categorical_dtype(events_df[col]):
            events_df[col] = events_df[col].cat.add_categories([99])

    mask = (
        events_df['LONGEST_JOB_DURATION'].notna() &
        (events_df['LONGEST_JOB_INDUSTRY'].isna() | events_df['LONGEST_JOB_OCCUPATION'].isna())
    )
    events_df.loc[mask, 'LONGEST_JOB_INDUSTRY'] = events_df.loc[mask, 'LONGEST_JOB_INDUSTRY'].fillna(99)
    events_df.loc[mask, 'LONGEST_JOB_OCCUPATION'] = events_df.loc[mask, 'LONGEST_JOB_OCCUPATION'].fillna(99)


    #WORKING HISTORY VARIABLES

    lh39_40_vars = ['LH39A', 'LH39B', 'LH39C', 'LH39D',  'LH40A', 'LH40B', 'LH40C', 'LH40D']

    events_df[lh39_40_vars] = events_df[lh39_40_vars].apply(pd.to_numeric, errors='coerce')

    mask = (
        (events_df[lh39_40_vars].isna().any(axis=1)) &
        ((events_df['LH38'] == 0) | (events_df['RAEVBRN'] == 0))
    )
    for var in lh39_40_vars:
        events_df.loc[mask & events_df[var].isna(), var] = 99

    for col in lh39_40_vars: 
        events_df[col] = events_df[col].replace([6], 99)




    def categorize_by_median(df, columns):
        """
        Categorizes specified columns as 1 (≤ median) or 2 (> median), keeps missing as missing.
        Adds new columns with '_cat' suffix.
        """
        df_cat = df.copy()
        for col in columns:
            median = df[col].median()
            df_cat[col + '_CAT'] = np.where(df[col].isna(), np.nan, np.where(df[col] > median, 2, 1))
        return df_cat




    events_df = does_not_apply(events_df, 'RAEDUC', [1, 2, 3], ['NUM_UNI'], 0)
    events_df = does_not_apply(events_df, 'RAEDUC', [1, 2, 3], ['PRIVATE_UNI'], 5)
    #drinks per week is 0 if they do not drink
    #'JOB_SATISFIED', 'SUPERVISOR_SUPPORT', 'COWORKER_SUPPORT','WORK_ENV', 'JOB_STRESS','NUM_UNI', 'PRIVATE_UNI',

    events_df.loc[
        (events_df['DAYS_WEEK_DRINKING'].isna()) & (events_df['CURRENTLY_DRINKS'] == 0),
        'DAYS_WEEK_DRINKING'
    ] = 0

    events_df['DAYS_WEEK_DRINKING'] = events_df['DAYS_WEEK_DRINKING'].replace((8, 9), pd.NA)
    events_df.loc[
        (events_df['DRINKS_PER_DAY'].isna()) & (events_df['CURRENTLY_DRINKS'] == 0),
        'DRINKS_PER_DAY'
    ] = 0

    events_df['DRINKS_PER_WEEK'] = events_df['DAYS_WEEK_DRINKING'] * events_df['DRINKS_PER_DAY']
    events_df['DRINKS_PER_WEEK_RAND'] = events_df['DAYS_WEEK_DRINKING_RAND'] * events_df['DRINKS_PER_DAY_RAND']
    ### FIXING LONGEST JOB INDUSTRY AND OCCUPATION
    industry_group = {
        1: "PRIMARY",
        2: "PRIMARY",
        3: "SECONDARY",
        4: "SECONDARY",
        5: "TERTIARY",
        6: "TERTIARY",
        7: "TERTIARY",
        8: "TERTIARY",
        9: "TERTIARY",
        10: "TERTIARY",
        11: "TERTIARY",
        12: "TERTIARY",
        13: "PUBLIC",
        99: "UNKNOWN"
    }

    occupation_group = {
        1: "WHITECOLLAR",
        2: "WHITECOLLAR",
        3: "WHITECOLLAR",
        4: "WHITECOLLAR",
        5: "SERVICE",
        6: "SERVICE",
        7: "SERVICE",
        8: "SERVICE",
        9: "SERVICE",
        10: "BLUECOLLAR",
        11: "BLUECOLLAR",
        12: "BLUECOLLAR",
        13: "BLUECOLLAR",
        14: "BLUECOLLAR",
        15: "BLUECOLLAR",
        16: "BLUECOLLAR",
        17: "ARMEDFORCES",
        99: "UNKNOWN"
    }

    events_df["LONGEST_JOB_INDUSTRY"] = events_df["LONGEST_JOB_INDUSTRY"].astype("Int64")
    events_df["LONGEST_JOB_OCCUPATION"] = events_df["LONGEST_JOB_OCCUPATION"].astype("Int64")

    events_df["LONGEST_INDUSTRY"] = events_df["LONGEST_JOB_INDUSTRY"].map(industry_group)
    events_df["LONGEST_OCCUPATION"] = events_df["LONGEST_JOB_OCCUPATION"].map(occupation_group)

    fjob_map = {
        1: "WhiteCollar",   # managerial/professional
        2: "WhiteCollar",   # sales
        3: "WhiteCollar",   # clerical
        4: "Service",
        5: "BlueCollar",
        6: "ArmedForces",
        8: np.nan,          # don't know
        9: np.nan           # not ascertained
    }

    events_df["FJOB"] = events_df["FJOB"].astype("Int64")
    events_df["FJOB_CAT"] = events_df["FJOB"].map(fjob_map)

    # one-hot encode the grouped categories
    events_df['NEVER_LIVED_FATHER'] = np.where(events_df['FAUNEM'] == 7, 1, np.where(events_df['FAUNEM'].isin([1, 5, 6]), 0, np.nan))
    events_df['NEVER_LIVED_MOTHER'] = np.where(events_df['MOWORK'] == 7, 1, np.where(events_df['MOWORK'].isin([1, 3, 5]), 0, np.nan))
    events_df['MOWORK'] = events_df['MOWORK'].replace({1: 1, 3: 2, 5: 3, 7: np.nan})
    events_df['FATHER_DISABLED'] = np.where(events_df['FAUNEM'] == 6, 1, np.where(events_df['FAUNEM'].isin([1, 5, 7]), 0, np.nan))
    events_df['FAUNEM'] = events_df['FAUNEM'].replace({1: 1, 5: 0, 6: 0, 7: 0})





    events_df = clean_lh1_data(events_df)





    #
    events_df['SPOKE_ENGLISH_HOME'] = np.where(events_df['LH12'].isin([1, 2, 3]), 1, np.where(events_df['LH12'] == 4, 0, np.nan))
    events_df['MULTILINGUAL_HOME'] = np.where(events_df['LH12'].isin([2, 3]), 1, np.where(events_df['LH12'].isin([1, 4]), 0, np.nan))


    events_df = events_df.drop(columns=['LH12'], errors='ignore')


    events_df = events_df.drop(columns=to_delete_dv.union(scores_to_aggregate_raws)).copy()















    events_df.to_csv(CLEAN_EVENTS_FILE, index=False)
    print("Done")


if __name__ == "__main__":
    main()
