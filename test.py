import requests
url = ('https://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=bea0d89d35684670ad8cf9e4293d5857')
response = requests.get(url)
print (response.json())


