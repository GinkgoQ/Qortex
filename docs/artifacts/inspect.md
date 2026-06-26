# Inspect Artifact

## Summary

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
print(art.summary())
```

Output:

```
Artifact: artifacts/ds004130/
Format:   parquet
Source:   ds004130@1.2.0
Created:  2024-01-15T14:23:00Z

Splits:
  train   1,200 samples   61 subjects   classes: rest(400) eyes-open(398) task(402)
  val       270 samples   14 subjects   classes: rest(90)  eyes-open(89)  task(91)
  test      270 samples   13 subjects   classes: rest(90)  eyes-open(88)  task(92)

Features: 491,520 (64 channels × 7,680 time points)
Window:   30.0 s, 0.5 overlap
```

## Peeking at a sample

```python
sample = art.peek(split="train", index=0)
print(sample.X.shape)       # (64, 7680)
print(sample.y)             # "rest"
print(sample.subject_id)    # "01"
print(sample.onset_s)       # 0.0
```

## File structure

```python
for split in ["train", "val", "test"]:
    shards = list(art.shard_paths(split))
    print(f"{split}: {len(shards)} shards")
```

## Checking artifact integrity

```python
ok = art.check_integrity()
# Reads each shard and verifies row counts match the manifest
print(ok)  # True / False
```

If integrity fails, specific shards are listed. Re-run conversion to fix.
