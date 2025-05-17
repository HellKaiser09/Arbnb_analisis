from pymongo import MongoClient
import pandas as pd
import numpy as np

cliente = MongoClient('mongodb://localhost:27017')
db = cliente["proyecto"]
coleccion = db["Airbnbs"]

datos = pd.read_csv("C:/Users/ultim/Downloads/proyecto/train.csv")

records = datos.to_dict(orient='records')
coleccion.insert_many(records)

print("Los datos se han importado correctamente")