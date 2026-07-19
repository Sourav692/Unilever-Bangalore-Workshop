# Decoding a docs.databricks.com example notebook

Databricks doc pages that host an example notebook (URLs like
`https://docs.databricks.com/aws/en/notebooks/source/.../some-notebook.html`) render the notebook
client-side. `WebFetch` sees only the page title. The real content is embedded as a
base64-encoded, then URL-encoded JSON blob assigned to `__DATABRICKS_NOTEBOOK_MODEL`.

To extract the actual cells:

```bash
cd /tmp
curl -sL "<NOTEBOOK_URL>" -o nb.html
# the blob is on one very long <script> line
python3 - <<'PY'
import re, base64, urllib.parse, json
data = open('/tmp/nb.html', encoding='utf-8', errors='replace').read()
m = re.search(r"__DATABRICKS_NOTEBOOK_MODEL = '([^']+)'", data)
model = json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode('utf-8')))
print("name:", model.get("name"), "| language:", model.get("language"))
for i, c in enumerate(model.get("commands", [])):
    print(f"\n===== CELL {i} =====\n{c.get('command','')}")
PY
```

Use the decoded cells to (a) verify your notebook covers the same concepts and (b) mirror the
official structure and API usage. Compare concept-by-concept and report gaps.
