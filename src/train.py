import os
import pickle
import pandas as pd
from NavieBayes import train_naive_bayes

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Số lượng giá trị thiếu mỗi cột:")
    print(df.isnull().sum())
    df = df.fillna(df.mode().iloc[0])
    features = list(df.columns[:-1])
    samples = df.values.tolist()
    return samples, features, df

if __name__ == "__main__":
    data_path = os.path.join('..', 'data', 'data.csv')
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại.")
        exit(1)

    data, features, df = load_data(data_path)
    print("Một vài dòng dữ liệu đầu tiên:")
    print(df.head())

    prior_prob, cond_prob = train_naive_bayes(data)

    # Lưu mô hình
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'naive_bayes_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump((prior_prob, cond_prob, features), f)
    print(f"Đã lưu mô hình vào {model_path}")
