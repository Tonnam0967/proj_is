import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# โหลดข้อมูลฝึก
data = pd.read_csv('hand_gesture_data.csv')

# แยกฟีเจอร์และ labels
X = data.drop('label', axis=1)
y = data['label']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ฝึกโมเดล
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

# ทำนายผลและประเมินความแม่นยำ
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# บันทึกโมเดล
import joblib
joblib.dump(model, 'hand_gesture_model.pkl')
joblib.dump(scaler, 'scaler.pkl')