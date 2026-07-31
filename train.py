from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow




def load_data(file_path):
    
    data = pd.read_csv(file_path)
    return data

df = load_data('data/fraud_cleaned.csv')


def prepared_data(df):

    y = df['__FeatEng_target__']
    X = df.drop('__FeatEng_target__', axis=1)
    train_x, test_x,train_y,test_y =train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return train_x, test_x, train_y, test_y 


def train_and_evaluate_model(train_x, test_x, train_y, test_y):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42,n_jobs=-1)
        model.fit(train_x, train_y)

        predictions = model.predict(test_x)
        accuracy = accuracy_score(test_y, predictions)
        report = classification_report(test_y, predictions,output_dict=True)
        mlflow.log_metric("fraud_precision",report['1']['precision'])
        mlflow.log_metric("fraud_recall", report['1']['recall'])
        mlflow.log_metric("fraud_f1",report['1']['f1-score'])

        mlflow.log_metric("accuracy", accuracy,)

    return accuracy, report


if __name__ == "__main__":
    df = load_data('data/fraud_cleaned.csv')
    train_x,test_x,train_y,test_y = prepared_data(df)
    accuracy, report = train_and_evaluate_model(train_x, test_x, train_y, test_y)
    print(accuracy)
    print(report)



