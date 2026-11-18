
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df_black_friday_sales = pd.read_csv('./BlackFriday.csv')

print(df_black_friday_sales.head())

print(df_black_friday_sales.isna().any())

print(df_black_friday_sales.isnull().sum())

df_black_friday_sales['Product_Category_2'] = df_black_friday_sales['Product_Category_2'].fillna(-2.0).astype("float32")

df_black_friday_sales['Product_Category_3'] = df_black_friday_sales['Product_Category_3'].fillna(-2.0).astype("float32")

df_black_friday_sales.fillna(df_black_friday_sales.mean(numeric_only=True), inplace=True)

print(df_black_friday_sales.head())

print(df_black_friday_sales[df_black_friday_sales.duplicated()].count().sum())

df = df_black_friday_sales

df_cleaned = df.drop_duplicates()

df_cleaned = df_cleaned.reset_index(drop=True)

scaler = MinMaxScaler()

df_black_friday_sales['Purchase'] = scaler.fit_transform(df_black_friday_sales[['Purchase']])

print(df_black_friday_sales.head())

age_mapping = {
    '0-17': 'toddler',
    '18-25': 'young_adult',
    '26-35': 'adult',
    '36-45': 'middle_age',
    '46-50': 'mid_age',
    '51-55': 'older_adult',
    '55+': 'elderly'
}

df_black_friday_sales['Age_new'] = df_black_friday_sales['Age'].map(age_mapping)

df_black_friday_sales = pd.get_dummies(df_black_friday_sales, columns=['Age_new'], drop_first=True)

cols = ['City_Category', 'Stay_In_Current_City_Years']

le = LabelEncoder()

for col in cols:
    df_black_friday_sales[col] = le.fit_transform(df_black_friday_sales[col])

df_black_friday_sales.head()

print("Dimensions of the dataset: ", df_black_friday_sales.shape, "\n\n")

df_black_friday_sales.info()

unique_values_gender = df_black_friday_sales['Gender'].unique()
unique_values_age = df_black_friday_sales['Age'].unique()
unique_values_city_category = df_black_friday_sales['City_Category'].unique()

print("\nUnique values for 'Gender':", unique_values_gender)
print("\nUnique values for 'Age':", unique_values_age)
print("\nUnique values for 'City_Category':", unique_values_city_category)

df_black_friday_sales['Gender'] = df_black_friday_sales['Gender'].replace({
    'M': 0,
    'F': 1
})

print(df_black_friday_sales['Gender'].head())

print(df_black_friday_sales.describe())

print("Data Types:\n")
print(df_black_friday_sales.dtypes)

unique_values = df_black_friday_sales['City_Category'].unique()

print("\nUnique Values in 'City_Category':")
print(unique_values)

sns.barplot(
    data=df_black_friday_sales,
    x='Gender',
    y='Purchase'
)

plt.title("Gender VS Purchase Distribution Plot")
plt.show()
