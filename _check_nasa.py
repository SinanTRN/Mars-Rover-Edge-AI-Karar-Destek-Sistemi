import requests

urls = [
    "https://data.nasa.gov/api/views/cykx-2qix",
    "https://data.nasa.gov/resource/cykx-2qix.json",
]

for url in urls:
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        print(f"{url} -> {r.status_code}")
    except Exception as e:
        print(f"{url} -> ERROR: {e}")
