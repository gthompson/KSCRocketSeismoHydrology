import requests
import pandas as pd
import datetime

# Example endpoint with filters (you might need to adjust parameters based on the API docs)
startdate='2016-01-01'
enddate=datetime.datetime.today().strftime('%Y-%m-%d')
#print(enddate)#'2025-12-31'
enddate='2020-11-25'
url = f"https://ll.thespacedevs.com/2.3.0/launches/previous/?search=cape&window_start__gte={startdate}&window_start__lte={enddate}&limit=100"
response = requests.get(url)
data = response.json().get('results', [])

# Convert the list of launches into a DataFrame
df = pd.DataFrame(data)

# Save DataFrame as CSV
df.to_csv('ksc_launches_2020.csv', index=False)
print("CSV file saved as 'ksc_launches.csv'")