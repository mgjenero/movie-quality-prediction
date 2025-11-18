import pickle
from sklearn.ensemble import RandomForestClassifier

import preprocessing


def train_model(df):
    X = df.drop(columns=['is_good'])
    y = df['is_good']


    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


def save_model(model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(model, f_out)

df = preprocessing.load_processed()
# df = preprocessing.load_raw()
model = train_model(df)
save_model(model, 'models/model.bin')

print('Model saved to model.bin')
