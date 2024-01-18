import requests

url = "https://tsaboin-tsaboin-cams-v1.p.rapidapi.com/listall.json"

headers = {
	"X-RapidAPI-Key": "77edeb4b2amsh26bc0277bad393dp117823jsn7451c2ea896e",
	"X-RapidAPI-Host": "tsaboin-tsaboin-cams-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(response.json())