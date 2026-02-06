import pandas as pd

# Load the old CSV file
df = pd.read_csv("Untitled form (Responses) - Form Responses 1-4.csv")

print("Before cleaning:")
print(df.head())
print(df.info())

# Drop unwanted columns
df = df.drop(columns=['Timestamp', 'Institution/college'])

# Function to clean study hours
def map_study_hours(x):
    if isinstance(x, str):
        x = x.lower()
        if '<' in x:
            return 2
        elif '5-10' in x or '5–10' in x:
            return 7.5
        elif '11-15' in x:
            return 13
        elif '>' in x or '15' in x:
            return 17
    return x

df['How many hours do you study per week on average?'] = \
df['How many hours do you study per week on average?'].apply(map_study_hours)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column mean
df = df.fillna(df.mean())

print("\nAfter cleaning:")
print(df.head())
print(df.info())

# Save cleaned CSV
df.to_csv("student_academic_cleaned.csv", index=False)

print("\n✅ Cleaned CSV saved as student_academic_cleaned.csv")
# Remove completely empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Save again (overwrite)
df.to_csv("student_academic_cleaned.csv", index=False)

print("✅ CSV fixed: empty columns removed")
