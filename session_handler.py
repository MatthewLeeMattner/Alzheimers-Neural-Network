import pandas as pd
import utils


class Subject:

    def __init__(self, subject, info_df, image_list):
        self.subject = subject
        self.info_df = info_df
        self._setup()

    def _setup(self):
        self.subject_rows = []
        [self.subject_rows.append(x) for _, x in self.info_df.iterrows()
            if x['Subject'] == self.subject]



if __name__ == "__main__":

    CSV_INFO = "/home/matthew-lee/Data/ADNI/2Yr_1.5T/" \
        "ADNI1_Complete_2Yr_1.5T_3_17_2019.csv"
    info_df = pd.read_csv(CSV_INFO)
    subject = Subject("941_S_1311", info_df)

    print(subject.subject_rows)
    print(len(subject.subject_rows))

    '''
    ADNI_{subject}
    '''
