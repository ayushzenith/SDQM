Version: 1.3 - at least one annotation for each synthetic image

# Running the Script

To run the script, use the following command:

```bash
python3 run.py [RarePlanes]
```

Ensure that `[RarePlanes]` is replaced with the path to the directory containing the "synthetic" and "real" subdirectories. The directory structure should look like this:

```text
RarePlanes/
├── synthetic/
│   ├── metadata_annotations/
│   ├── test/
│   └── train/
└── real/
    ├── metadata_annotations/
    ├── test/
    └── train/
```