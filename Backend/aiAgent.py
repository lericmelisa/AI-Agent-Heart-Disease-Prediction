import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from flask import Flask, request, jsonify
from flask_cors import CORS
import plotly.graph_objects as go
from joblib import load
from flask_cors import cross_origin
from sklearn.metrics import precision_score

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Function to load the dataset from a CSV file
def ucitaj_podatke(file_path="Heart_disease_cleveland_new.csv"):
    podaci = pd.read_csv(file_path)
    return podaci

# Function to process and split the dataset
def pripremi_podatke_za_treniranje(podaci):
    # Selecting the features (age, cp,chol,thalach,oldpeak) 
    # and the target variable (heart disease diagnosis)
    X = podaci[['age', 'cp','chol', 'thalach', 'oldpeak']]
    y = podaci['target']  # 'target' column indicates 
    #heart disease (1 = disease, 0 = no disease)

    # Split the dataset into training and test sets (80% train, 20% test)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the models
def treniraj_model(X_train, y_train,X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Izračunavanje preciznosti
    preciznost = precision_score(y_test, y_pred)
    print(f"Preciznost modela: {preciznost:.2f}")
    accuracy = model.score(X_test, y_test)
    print(f"Tačnost modela: {accuracy}")
    # Save the trained model
    dump(model, 'Heart_disease_cleveland_new.joblib')

# Function to predict heart disease based on input features
def predvidi_hoce_li_imati_srcanu_bolest(age,cp,chol,thalach,oldpeak):
    ulazne_karakteristike = pd.DataFrame([{
        'age': age,
        'cp': cp,
        'chol': chol,
        'thalach': thalach,
        'oldpeak': oldpeak
    }])

    # Load the pre-trained model
    model = load('Heart_disease_cleveland_new.joblib')
    probabilities = model.predict_proba(ulazne_karakteristike)[0]
    prediction = model.predict(ulazne_karakteristike)[0]
    
    return prediction, probabilities


def dodaj_nove_podatke(novi_podaci, file_path="Heart_disease_cleveland_new.csv"):
    # Checking if the new data contains all required columns
    required_columns = {'age', 'cp', 'chol', 'thalach', 'oldpeak'}
    if not required_columns.issubset(novi_podaci.columns):
        raise ValueError(f"Novi podaci moraju sadrzavati potrebne atribute: {required_columns}")

    # Load the existing data
    data = ucitaj_podatke(file_path)

    # Append the new data to the existing data
    data = pd.concat([data,novi_podaci], ignore_index=True)

    # Save the updated data to a CSV file
    data.to_csv(file_path, index=False) 



@app.route('/predvidi', methods=['POST'])
def predvidi():
    try:
        # Check if the request contains JSON data
        sadrzaj = request.json

        # List of required keys
        required_keys = ['age', 'cp', 'chol', 'thalach', 'oldpeak']
        
        # Check if all required keys are present
        for key in required_keys:
            if key not in sadrzaj:
                return jsonify({'error': f"Missing required field: {key}"}), 400
        
        
        try:
            age = (sadrzaj['age'])
            cp = (sadrzaj['cp'])
            chol = (sadrzaj['chol'])
            thalach = (sadrzaj['thalach'])
            oldpeak = (sadrzaj['oldpeak'])
        except ValueError as e:
            return jsonify({'error': f"Invalid data type: {str(e)}"}), 400
        
        # Make a prediction
        prediction, probabilities = predvidi_hoce_li_imati_srcanu_bolest(age, cp, chol, thalach, oldpeak)
        
        
        sadrzaj['target'] = prediction  # Adding the prediction (target) to the data

        # Spremanje novih podataka u CSV
        novi_podaci = pd.DataFrame([sadrzaj])
        dodaj_nove_podatke(novi_podaci)

        # Vraćanje rezultata predikcije
        return jsonify({
            'Predikcija': 'Pacijent ima srčanu bolest' if prediction == 1 else 'Pacijent nema srčanu bolest',
            'Pouzdanost': {
            'Ima srčanu bolest': f"{probabilities[1] * 100:.2f}%",
            'Nema srčanu bolest': f"{probabilities[0] * 100:.2f}%"
            }
        })

    except Exception as e:
        # U slučaju bilo koje druge greške, šaljemo korisniku informaciju o grešci
     return jsonify({'error': f"Server error: {str(e)}"}), 500
    


@app.route('/retrain_model', methods=['POST'])
@cross_origin()  # Allow cross-origin requests
def train_again_endpoint():
    try:
        # Load the data for retraining
        print("Učitavanje podataka za ponovno treniranje...")
        data = ucitaj_podatke("Heart_disease_cleveland_new.csv")

        # Process and split the data for training
        print("Obrada i podjela podataka...")
        X_train, X_test, y_train, y_test = pripremi_podatke_za_treniranje(data)

        # Retrain the model
        print("Ponovno treniranje modela...")
        treniraj_model(X_train, y_train,X_test, y_test) # Train the model   

        print("Model uspješno ponovo obučen.")
        return jsonify({'message': 'Model uspješno ponovo obučen.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        # Load the model
        model = load('Heart_disease_cleveland_new.joblib') 
        data = ucitaj_podatke('Heart_disease_cleveland_new.csv')  # Load dataset

        # Select the features and labels
        features = data[['age', 'cp', 'chol', 'thalach', 'oldpeak']]
        labels = data['target']

        # Make predictions
        predictions = model.predict(features)
        

        # Create 3D scatter plot
        fig = go.Figure()

        # Add data for positive cases (target = 1)
        positive = features[labels == 1]
        fig.add_trace(
            go.Scatter3d(
                x=positive['age'],
                y=positive['cp'],
                z=positive['thalach'],  
                mode='markers',
                marker=dict(
                    size=positive['oldpeak'] * 3,  # Scale size based on oldpeak
                    color=positive['chol'],  # Cholesterol intensity
                    colorscale='Bluered',  # Gradient for cholesterol
                    opacity=0.8,
                    colorbar=dict(title="Chol")  # Add colorbar for cholesterol
                ),
                text=[f"Age: {age}, CP: {cp}, Chol: {chol}, Thalach: {thalach}<br>"
                      f"Oldpeak: {oldpeak}, Target: 1, Predicted: {pred}" 
                      for age, cp, chol, thalach, oldpeak, pred in zip(
                          positive['age'], 
                          positive['cp'], 
                          positive['chol'], 
                          positive['thalach'], 
                          positive['oldpeak'], 
                          predictions[labels == 1]
                        
                      )]
            )
        )

        # Add data for negative cases (target = 0)
        negative = features[labels == 0]
        fig.add_trace(
            go.Scatter3d(
                x=negative['age'],
                y=negative['cp'],
                z=negative['thalach'],
                mode='markers',
                marker=dict(
                    size=negative['oldpeak'] * 3,  # Scale size based on oldpeak
                    color=negative['chol'],  # Cholesterol intensity
                    colorscale='Bluered',  # Gradient for cholesterol
                    opacity=0.8
                ),
                text=[f"Age: {age}, CP: {cp}, Chol: {chol}, Thalach: {thalach}<br>"
                      f"Oldpeak: {oldpeak}, Target: 0, Predicted: {pred}" 
                      for age, cp, chol, thalach, oldpeak, pred in zip(
                          negative['age'], 
                          negative['cp'], 
                          negative['chol'], 
                          negative['thalach'], 
                          negative['oldpeak'], 
                          predictions[labels == 0]
                          
                      )]
            )
        )
        # Update layout for the plot
        fig.update_layout(
            title='Interaktivni 3D scatter plot',
            scene=dict(
                xaxis_title='Age',  
                yaxis_title='CP',  
                zaxis_title='Thalach',  
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(
                x=0.05,  # Move legend outside the plot
                y=1,
                title=""
            )     )
        # Convert Plotly figure to JSON-compatible format
        graph_json = fig.to_json()
        return jsonify({'graph': graph_json}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



    
@app.route('/feature_importances', methods=['GET'])
@cross_origin()  # Allow cross-origin requests
def feature_importances():
    try:
        # Load the trained model
        model = load('Heart_disease_cleveland_new.joblib')

        # Extract feature importances and convert to a Python list
        feature_importances = model.feature_importances_.tolist()
        features = ['age', 'cp', 'chol', 'thalach', 'oldpeak']

        # Create a Plotly figure
        fig = go.Figure(data=[
            go.Bar(name='Importance', x=features, y=feature_importances)
        ])
        fig.update_layout(
            title='Važnost atributa',
            xaxis_title='Atributi',
            yaxis_title='Važnost',
            template='plotly_dark'
        )

        # Convert Plotly figure to a JSON-compatible format
        return jsonify(fig.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)



