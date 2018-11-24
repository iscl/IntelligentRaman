from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
print(le.fit([1, 5, 67, 100]))
print(le.transform([1, 1, 100, 67, 5]))