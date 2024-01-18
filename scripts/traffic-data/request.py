import requests

url = "https://api.helios.earth/v1/cameras?bbox=-77.636,43.142,-77.582,43.171"



response = requests.get(url)

print(response.json())