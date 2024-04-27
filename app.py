import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open(os.path.join('data', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('data', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('data', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions

def check_request(request, response):
    if "data" not in request:
        error = "data error"
        return False, error
    return True, ""


def check_valid_column(observation):
    valid_columns = {"age", "sex", "race", "workclass", "education", "marital-status", "capital-gain",
                     "capital-loss", "hours-per-week"}
    for key in observation.keys():
        if key not in valid_columns:
            error = f"{key} not in columns"
            return False, error
    for col in valid_columns:
        if col not in observation.keys():
            error = f"{col} not in columns"
            return False, error
    return True, ""


def check_categorical_values(observation):
    valid_category = {'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov',
                                    '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
                      'education': ['Bachelors', 'HS-grad','11th','Masters','9th','Some-college',
                                    'Assoc-acdm','Assoc-voc','7th-8th', 'Doctorate','Prof-school',
                                    '5th-6th','10th','1st-4th','Preschool','12th'],
                      'marital-status': ['Never-married','Married-civ-spouse','Divorced',
                                         'Married-spouse-absent','Separated','Married-AF-spouse','Widowed'],
                      'race': ['White','Black','Asian-Pac-Islander','Amer-Indian-Eskimo','Other'],
                      'sex': ['Male', 'Female']}

    for key, valid in valid_category.items():
        if key in observation:
            value = observation[key]
            if value not in valid_category[key]:
                error = f"{value} not valid for {key}"
                return False, error 
        else:
            error = "error"
            return False, error 
    return True, ""



def check_numericals(observation, response):
    age = observation.get("age")
    if not isinstance(age, int) or age > 100 or age < 0: 
        error = f"age {age} not valid"
        return False, error 
    capital_gain = observation.get("capital-gain")
    if not isinstance(capital_gain, int) or capital_gain < 0: 
        error = f"capital-gain {capital_gain} not valid"
        return False, error
    capital_loss = observation.get("capital-loss")
    if not isinstance(capital_loss, int) or capital_loss < 0: 
        error  = f"capital-loss {capital_loss} not valid"
        return False, error
    hour = observation.get("hours-per-week")
    if not isinstance(hour, int) or hour < 0 or hour >  168: 
        error = f" hours-per-week {hour} not valid"
        return False, error 
    return True, ""


# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
  
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'error': error}
        return jsonify(response)
    
    if 'observation_id' in obs_dict:
        _id = str(obs_dict['observation_id'])
        response = {}
        response["observation_id"] = _id
    else:
        _id = None
        error = "observation_id error"
        response = {'error': error}
        return jsonify(response)

    observation = obs_dict['data']

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    nums, error = check_numericals(observation)
    if not nums:
        response = {'error': error}
        return jsonify(response)

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response["prediction"] = prediction
    response["probability"] = proba
    
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
        
    return jsonify(response)

    
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
