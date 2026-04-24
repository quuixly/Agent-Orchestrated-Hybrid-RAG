import pandas as pd


data = pd.read_json("data/medical-exams-LEK-PL-2008-2024.json")
data = data.sample(frac=1)

sample = data.sample(400)
sample.to_json("data/lek_pl_sample.json")