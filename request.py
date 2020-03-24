import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'text':'give it to me'})

print(r.json())
