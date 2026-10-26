from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data_small"
    output.mkdir(exist_ok=True)
    stories = [
        "Lina found a red ball in the garden. She rolled it to her brother. They laughed and played until sunset.\n",
        "A small fox walked through the forest. It heard a bird and stopped to listen. Then it went home safely.\n",
        "Tom wanted to build a boat. His father gave him wood and a hammer. Together they made a boat that floated.\n",
        "Mia planted a seed in a cup. She gave it water every day. A green leaf appeared and Mia felt proud.\n",
        "The little dog was cold. Ava wrapped it in a warm blanket. The dog wagged its tail and fell asleep.\n",
    ]
    train = "".join(stories * 120)
    valid = "".join(stories * 12)
    (output / "train.txt").write_text(train, encoding="utf-8")
    (output / "valid.txt").write_text(valid, encoding="utf-8")


if __name__ == "__main__":
    main()
