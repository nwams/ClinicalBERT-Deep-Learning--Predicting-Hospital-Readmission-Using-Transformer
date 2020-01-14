import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load ADMISSIONS table
df_adm = pd.read_csv(
    '/Users/nwams/Documents/Machine Learning Projects/Predicting-Hospital-Readmission-using-NLP/ADMISSIONS.csv')

'''
Convert Strings to Dates.
When converting dates, it is safer to use a datetime format. 
Setting the errors = 'coerce' flag allows for missing dates 
but it sets it to NaT (not a datetime)  when the string doesn't match the format.
'''
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')

'''
Get the next Unplanned admission date for each patient (if it exists).
I need to get the next admission date, if it exists.
First I'll verify that the dates are in order.
Then I'll use the shift() function to get the next admission date.
'''
df_adm = df_adm.sort_values(['SUBJECT_ID', 'ADMITTIME'])
df_adm = df_adm.reset_index(drop=True)
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

'''
Since I want to predict unplanned re-admissions I will drop (filter out) any future admissions that are ELECTIVE 
so that only EMERGENCY re-admissions are measured.
For rows with 'elective' admissions, replace it with NaT and NaN
'''
rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

# It's safer to sort right before the fill incase something I did above changed the order
df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

'''
Backfill in the values that I removed. So copy the ADMITTIME from the last emergency 
and paste it in the NEXT_ADMITTIME for the previous emergency. 
So I am effectively ignoring/skipping the ELECTIVE admission row completely. 
Doing this will allow me to calculate the days until the next admission.
'''
# Back fill. This will take a little while.
df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

# Calculate days until next admission
df_adm['DAYS_TIL_NEXT_ADMIT'] = (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)

'''
Remove NEWBORN admissions
According to the MIMIC site "Newborn indicates that the HADM_ID pertains to the patient's birth."

I will remove all NEWBORN admission types because in this project I'm not interested in studying births — my primary 
interest is EMERGENCY and URGENT admissions.
I will remove all admissions that have a DEATHTIME because in this project I'm studying re-admissions, not mortality. 
And a patient who died cannot be re-admitted.
'''
df_adm = df_adm.loc[df_adm.ADMISSION_TYPE != 'NEWBORN']
df_adm = df_adm.loc[df_adm.DEATHTIME.isnull()]

'''
Make Output Label
For this problem, we are going to classify if a patient will be admitted in the next 30 days. 
Therefore, we need to create a variable with the output label (1 = readmitted, 0 = not readmitted).
'''
df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')


# Load NOTEEVENTS Table
df_notes = pd.read_csv("/Users/nwams/Documents/Machine Learning Projects/Predicting-Hospital-Readmission-using-NLP/NOTEEVENTS.csv")

# Sort by subject_ID, HAD_ID then CHARTDATE
df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
# Merge notes table to admissions table
df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','OUTPUT_LABEL','DURATION']],
                        df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY']],
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')

# Grab date only, not the time
df_adm_notes.ADMITTIME_C = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])

df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')

# Gather Discharge Summaries Only
df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
# multiple discharge summary for one admission -> after examination -> replicated summary -> replace with the last one
df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
df_discharge=df_discharge[df_discharge['TEXT'].notnull()]

### If Less than n days on admission notes (Early notes)
def less_n_days_data(df_adm_notes, n):
    df_less_n = df_adm_notes[
        ((df_adm_notes['CHARTDATE'] - df_adm_notes['ADMITTIME_C']).dt.total_seconds() / (24 * 60 * 60)) < n]
    df_less_n = df_less_n[df_less_n['TEXT'].notnull()]
    # concatenate first
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(
        lambda x: df_less_n[df_less_n['HADM_ID'] == x].OUTPUT_LABEL.values[0])

    return df_concat

df_less_2 = less_n_days_data(df_adm_notes, 2)
df_less_3 = less_n_days_data(df_adm_notes, 3)

import re

def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y

def preprocessing(df_less_n):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()

    df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))

    # to get 318 words chunks for readmission tasks
    from tqdm import tqdm
    df_len = len(df_less_n)
    want = pd.DataFrame({'ID': [], 'TEXT': [], 'Label': []})
    for i in tqdm(range(df_len)):
        x = df_less_n.TEXT.iloc[i].split()
        n = int(len(x) / 318)
        for j in range(n):
            want = want.append({'TEXT': ' '.join(x[j * 318:(j + 1) * 318]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                                'ID': df_less_n.HADM_ID.iloc[i]}, ignore_index=True)
        if len(x) % 318 > 10:
            want = want.append({'TEXT': ' '.join(x[-(len(x) % 318):]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                                'ID': df_less_n.HADM_ID.iloc[i]}, ignore_index=True)

    return want


df_discharge = preprocessing(df_discharge)
df_less_2 = preprocessing(df_less_2)
df_less_3 = preprocessing(df_less_3)

### An example to get the train/test/split with random state:
### note that we divide on patient admission level and share among experiments, instead of notes level.
### This way, since our methods run on the same set of admissions, we can see the
### progression of readmission scores.

readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 1].HADM_ID
not_readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 0].HADM_ID
# subsampling to get the balanced pos/neg numbers of patients for each dataset
not_readmit_ID_use = not_readmit_ID.sample(n=len(readmit_ID), random_state=1)
id_val_test_t = readmit_ID.sample(frac=0.2, random_state=1)
id_val_test_f = not_readmit_ID_use.sample(frac=0.2, random_state=1)

id_train_t = readmit_ID.drop(id_val_test_t.index)
id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)

id_val_t = id_val_test_t.sample(frac=0.5, random_state=1)
id_test_t = id_val_test_t.drop(id_val_t.index)

id_val_f = id_val_test_f.sample(frac=0.5, random_state=1)
id_test_f = id_val_test_f.drop(id_val_f.index)

# test if there is overlap between train and test, should return "array([], dtype=int64)"
(pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values

id_test = pd.concat([id_test_t, id_test_f])
test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
                             columns=['id', 'label'])

id_val = pd.concat([id_val_t, id_val_f])
val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))), columns=['id', 'label'])

id_train = pd.concat([id_train_t, id_train_f])
train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
                              columns=['id', 'label'])

# get discharge train/val/test

discharge_train = df_discharge[df_discharge.ID.isin(train_id_label.id)]
discharge_val = df_discharge[df_discharge.ID.isin(val_id_label.id)]
discharge_test = df_discharge[df_discharge.ID.isin(test_id_label.id)]

# subsampling for training....since we obtain training on patient admission level so now we have same number of pos/neg readmission
# but each admission is associated with different length of notes and we train on each chunks of notes, not on the admission, we need
# to balance the pos/neg chunks on training set. (val and test set are fine) Usually, positive admissions have longer notes, so we need
# find some negative chunks of notes from not_readmit_ID that we haven't used yet

df = pd.concat([not_readmit_ID_use, not_readmit_ID])
df = df.drop_duplicates(keep=False)
# check to see if there are overlaps
(pd.Index(df).intersection(pd.Index(not_readmit_ID_use))).values

# for this set of split with random_state=1, we find we need 400 more negative training samples
not_readmit_ID_more = df.sample(n=400, random_state=1)
discharge_train_snippets = pd.concat([df_discharge[df_discharge.ID.isin(not_readmit_ID_more)], discharge_train])

# shuffle
discharge_train_snippets = discharge_train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)

# check if balanced
discharge_train_snippets.Label.value_counts()

discharge_train_snippets.to_csv('./discharge/train.csv')
discharge_val.to_csv('./discharge/val.csv')
discharge_test.to_csv('./discharge/test.csv')

### for Early notes experiment: we only need to find training set for 3 days, then we can test
### both 3 days and 2 days. Since we split the data on patient level and experiments share admissions
### in order to see the progression, the 2 days training dataset is a subset of 3 days training set.
### So we only train 3 days and we can test/val on both 2 & 3days or any time smaller than 3 days. This means
### if we train on a dataset with all the notes in n days, we can predict readmissions smaller than n days.

# for 3 days note, similar to discharge

early_train = df_less_3[df_less_3.ID.isin(train_id_label.id)]
not_readmit_ID_more = df.sample(n=500, random_state=1)
early_train_snippets = pd.concat([df_less_3[df_less_3.ID.isin(not_readmit_ID_more)], early_train])
# shuffle
early_train_snippets = early_train_snippets.sample(frac=1, random_state=1).reset_index(drop=True)
early_train_snippets.to_csv('./3days/train.csv')

early_val = df_less_3[df_less_3.ID.isin(val_id_label.id)]
early_val.to_csv('./3days/val.csv')

# we want to test on admissions that are not discharged already. So for less than 3 days of notes experiment,
# we filter out admissions discharged within 3 days
actionable_ID_3days = df_adm[df_adm['DURATION'] >= 3].HADM_ID
test_actionable_id_label = test_id_label[test_id_label.id.isin(actionable_ID_3days)]
early_test = df_less_3[df_less_3.ID.isin(test_actionable_id_label.id)]

early_test.to_csv('./3days/test.csv')

# for 2 days notes, we only obtain test set. Since the model parameters are tuned on the val set of 3 days

actionable_ID_2days = df_adm[df_adm['DURATION'] >= 2].HADM_ID

test_actionable_id_label_2days = test_id_label[test_id_label.id.isin(actionable_ID_2days)]

early_test_2days = df_less_2[df_less_2.ID.isin(test_actionable_id_label_2days.id)]

early_test_2days.to_csv('./2days/test.csv')