import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

class HousePricePredictor:
    def __init__(self, data_path):
        self.dataset = pd.read_excel(data_path)
        self.model = None
        self.preprocess_data()

    def preprocess_data(self):
        # Remove 'Id' column and handle missing values
        self.dataset.drop(['Id'], axis=1, inplace=True)
        self.dataset['SalePrice'] = self.dataset['SalePrice'].fillna(self.dataset['SalePrice'].mean())
        new_dataset = self.dataset.dropna()

        # One-Hot Encoding
        object_cols = new_dataset.select_dtypes(include='object').columns
        OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
        OH_cols.index = new_dataset.index
        OH_cols.columns = OH_encoder.get_feature_names_out()
        self.df_final = pd.concat([new_dataset.drop(object_cols, axis=1), OH_cols], axis=1)

        # Split the dataset into features and target
        self.X = self.df_final.drop('SalePrice', axis=1)
        self.Y = self.df_final['SalePrice']

        # Train/test split
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.X, self.Y, train_size=0.8, test_size=0.2, random_state=0)

    def train_model(self, model_type='SVR'):
        if model_type == 'SVR':
            self.model = SVR()
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=10)
        elif model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif model_type == 'CatBoost':
            self.model = CatBoostRegressor()
        else:
            raise ValueError("Unsupported model type")

        # Train the selected model
        self.model.fit(self.X_train, self.Y_train)

    def predict(self, input_data):
        # Ensure the input_data matches the training data structure
        if isinstance(input_data, pd.DataFrame):
            return self.model.predict(input_data)
        else:
            raise ValueError("Input data must be a pandas DataFrame")

    def evaluate(self):
        # Evaluate the model using MAPE
        predictions = self.model.predict(self.X_valid)
        if isinstance(self.model, CatBoostRegressor):
            print('R2 Score:', r2_score(self.Y_valid, predictions))
        else:
            print('MAPE:', mean_absolute_percentage_error(self.Y_valid, predictions))

# Usage
data_path = "HousePricePrediction.xlsx"  # Path to your data
predictor = HousePricePredictor(data_path)


predictor.train_model(model_type='LinearRegression')  

columns = predictor.X_train.columns
print(columns)

# Evaluate the model
predictor.evaluate()

# Implementation
new_home_data = pd.DataFrame([{
    'MSSubClass': 60,           
    'LotArea': 8500,            
    'OverallCond': 5,           
    'YearBuilt': 2005,          
    'YearRemodAdd': 2007,       
    'BsmtFinSF2': 0,            
    'TotalBsmtSF': 1000,        
    'MSZoning_C (all)': 0,      
    'MSZoning_FV': 0,
    'MSZoning_RH': 0,
    'MSZoning_RL': 1,           
    'MSZoning_RM': 0,
    'LotConfig_Corner': 0,
    'LotConfig_CulDSac': 0,
    'LotConfig_FR2': 0,
    'LotConfig_FR3': 0,
    'LotConfig_Inside': 1,      
    'BldgType_1Fam': 1,         
    'BldgType_2fmCon': 0,
    'BldgType_Duplex': 0,
    'BldgType_Twnhs': 0,
    'BldgType_TwnhsE': 0,
    'Exterior1st_AsbShng': 0,
    'Exterior1st_AsphShn': 0,
    'Exterior1st_BrkComm': 0,
    'Exterior1st_BrkFace': 0,
    'Exterior1st_CBlock': 0,
    'Exterior1st_CemntBd': 0,
    'Exterior1st_HdBoard': 0,
    'Exterior1st_ImStucc': 0,
    'Exterior1st_MetalSd': 0,
    'Exterior1st_Plywood': 0,
    'Exterior1st_Stone': 0,
    'Exterior1st_Stucco': 0,
    'Exterior1st_VinylSd': 1,   
    'Exterior1st_Wd Sdng': 0,
    'Exterior1st_WdShing': 0,
}], columns=predictor.X_train.columns)

predicted_price = predictor.predict(new_home_data)
print(f"Predicted Price: {predicted_price[0].round(2)}")