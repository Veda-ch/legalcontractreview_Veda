import json

with open(
    "data/knowledge_base/legal_records.jsonl",
    "r",
    encoding="utf-8"
) as f:

    records = [
        json.loads(line)
        for line in f
    ]

for source in ["CUAD", "MAUD"]:
    print("\n" + "=" * 60)
    print(source)
    print("=" * 60)

    count = 0

    for record in records:
        if record["source"] == source:
            print(json.dumps(record, indent=2, ensure_ascii=False))
            count += 1

            if count == 5:
                break