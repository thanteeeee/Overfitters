import pandas as pd
import re
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
class dataloading:
    def __init__(self, kaggle_path , survey_path):
        self.kaggle_path = kaggle_path
        self.survey_path = survey_path

    def load_data(self):
        df = pd.read_csv(self.kaggle_path)
        df2 = pd.read_excel(self.survey_path)
        return df, df2
    
    def change_column_names(self, df, df2):
        df2.drop(df2.columns[0],axis=1, inplace=True)
        df2.columns = df.columns

    def merge_data(self, df, df2):
        combined = pd.concat([df, df2], ignore_index=True)
        return combined
class cleaning:
    def __init__(self, combined):
        self.combined = combined

    def drop_duplicates(self):
        self.combined.drop_duplicates(inplace=True)

    def handle_missing_values(self):
        for col in self.combined.columns:
            if self.combined[col].dtype == 'object':
                self.combined[col] = self.combined[col].fillna(self.combined[col].mode()[0])
            elif self.combined[col].dtype in ['int64', 'float64']:
                self.combined[col] = self.combined[col].fillna(self.combined[col].mean())

    def inconsistent_data(self):
        self.combined["Feel Rested"] = self.combined["Feel Rested"].replace("ไม่สดชื่น", "No").replace("สดชื่น", "Yes")
        self.combined["Use Before Sleep"] = self.combined["Use Before Sleep"].replace("ไม่ใช่", "No").replace("ใช่", "Yes")
        self.combined["Anxiety/Low Mood"] = self.combined["Anxiety/Low Mood"].replace("ไม่หงุดหงิด/ไม่วิตกกังวล", "No").replace("หงุดหงิด/วิตกกังวล", "Yes")
        self.combined["Wellness Apps"] = self.combined["Wellness Apps"].replace("ไม่ใช่", "No").replace("ใช่", "Yes")
        self.combined["Sleep Quality"] = self.combined["Sleep Quality"].replace("ไม่ดี", "Bad").replace("ดี", "Good")
        self.combined["Screen Time Affects Sleep?"] = self.combined["Screen Time Affects Sleep?"].replace("ไม่แน่ใจ", "Not Sure").replace("ใช่", "Yes").replace("ไม่ใช่", "No")
    
    def extract_number(self, value):
        value = str(value)
        if "-" in value:
            cleaned = re.sub(r'[^\d\-.]', '', value)
            values = cleaned.split("-")
            if len(values) >= 2:
                return f"{values[0]}-{values[1]}"
        match = re.search(r'\d+', value)
        if match:
            return match.group()
        return None
    
    def clean_range_with_mean(self, value):
        if isinstance(value, str):
            if '-' in value:
                try:
                    low, high = map(float, value .split('-'))
                    return (low + high) / 2
                except ValueError:
                    return value
            else:
                try:
                    return float(value)
                except ValueError:
                    return value
        else:
            return value

    def apply_cleaning(self):
        self.combined["Age"] = self.combined["Age"].apply(self.extract_number)
        self.combined["Age"] = self.combined["Age"].astype(float)
        self.combined["Sleep Hours"] = self.combined["Sleep Hours"].apply(self.extract_number)
        self.combined["Sleep Hours"] = self.combined["Sleep Hours"].apply(self.clean_range_with_mean)
        self.combined["Daily Screen Time"] = self.combined["Daily Screen Time"].apply(self.extract_number)
        self.combined["Daily Screen Time"] = self.combined["Daily Screen Time"].apply(self.clean_range_with_mean)
    
    def handle_outliers(self):
        self.combined['Daily Screen Time'] = winsorize(self.combined['Daily Screen Time'].to_numpy(),(0.1, 0.1))
        self.combined['Age'] = winsorize(self.combined['Age'].to_numpy(),(0.1, 0.1))

class transformation:
    def __init__(self, combined):
        self.combined = combined

    def encode_categorical(self):
        binary_columns = [ "Use Before Sleep", "Anxiety/Low Mood", "Wellness Apps"]
        for col in binary_columns:
            self.combined[col] = self.combined[col].map({"Yes": 1, "No": 0})
        self.combined["Feel Rested"] = self.combined["Feel Rested"].map({"Yes": 2, "Sometimes": 1, "No": 0})
        self.combined["Sleep Quality"] = self.combined["Sleep Quality"].map({"Good": 1, "Bad": 0})
        self.combined["Screen Time Affects Sleep?"] = self.combined["Screen Time Affects Sleep?"].map({"Yes": 1, "No": 0, "Not Sure": 2})
    
    def standardscale(self):
        scaler = StandardScaler()
        numeric_cols = [col for col in self.combined.select_dtypes(include=['int64', 'float64']).columns if col not in ['Screen Time Affects Sleep?']]
        self.combined[numeric_cols] = scaler.fit_transform(self.combined[numeric_cols])


class pipeline:
    def __init__(self, kaggle_path, survey_path):
        self.kaggle_path = kaggle_path
        self.survey_path = survey_path

    def run_pipeline(self):
        data_loader = dataloading(self.kaggle_path, self.survey_path)
        df, df2 = data_loader.load_data()
        data_loader.change_column_names(df, df2)
        combined = data_loader.merge_data(df, df2)

        cleaner = cleaning(combined)
        cleaner.inconsistent_data() 
        cleaner.drop_duplicates()
        cleaner.handle_missing_values()
        cleaner.apply_cleaning()
        cleaner.handle_outliers()

        transformer = transformation(cleaner.combined)
        transformer.encode_categorical()
        transformer.standardscale()
        
        transformer.combined.to_csv("dataset/final_data.csv", index=False)
        return transformer.combined
    
if __name__ == "__main__":
    pipeline_instance = pipeline('dataset/kgdataset.csv', 'dataset/survey_data.xlsx')
    final_data = pipeline_instance.run_pipeline()
    print(final_data.head())