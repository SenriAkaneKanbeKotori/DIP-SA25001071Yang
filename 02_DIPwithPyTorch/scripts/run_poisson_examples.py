from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_blending_gradio import blending  # noqa: E402


EXAMPLES = [
    {
        "name": "water",
        "source": ROOT / "data_poisson" / "water" / "source.jpg",
        "target": ROOT / "data_poisson" / "water" / "target.jpg",
        "points": [(95, 50), (288, 42), (330, 178), (211, 241), (75, 194)],
        "dx": 85,
        "dy": 80,
    },
    {
        "name": "monolisa",
        "source": ROOT / "data_poisson" / "monolisa" / "source.png",
        "target": ROOT / "data_poisson" / "monolisa" / "target.png",
        "points": [(96, 91), (319, 86), (350, 370), (206, 456), (63, 372)],
        "dx": 0,
        "dy": 0,
    },
    {
        "name": "equation",
        "source": ROOT / "data_poisson" / "equation" / "source.png",
        "target": ROOT / "data_poisson" / "equation" / "target.png",
        "points": [(31, 37), (391, 31), (397, 358), (35, 361)],
        "dx": 1190,
        "dy": 615,
    },
]


def main():
    output_dir = ROOT / "assets" / "poisson_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for example in EXAMPLES:
        source = Image.open(example["source"])
        target = Image.open(example["target"])
        polygon_state = {"points": example["points"], "closed": True}
        result = blending(
            source,
            target,
            example["dx"],
            example["dy"],
            polygon_state,
            iter_num=600,
        )
        Image.fromarray(result).save(output_dir / f"{example['name']}_result.png")
        print(f"Saved {output_dir / (example['name'] + '_result.png')}")


if __name__ == "__main__":
    main()
