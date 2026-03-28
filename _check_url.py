import requests
r = requests.head(
    "https://zenodo.org/records/4560762/files/ai4mars-dataset-merged-0.4.zip",
    allow_redirects=True, timeout=15
)
print(f"Status: {r.status_code}")
cl = r.headers.get("content-length", "0")
print(f"Size: {int(cl)/1024/1024:.0f} MB")
