#!/usr/bin/env python3
"""Generate Q&A dataset for evaluation."""
import json
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Q&A templates by difficulty
FACTUAL_EASY = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Western Europe. Its capital and largest city is Paris, known for the Eiffel Tower and the Louvre Museum.",
        "answer": "Paris",
        "reference_facts": ["Paris is the capital of France", "Paris is the largest city in France"],
        "slice": ["factual", "easy", "geography"]
    },
    {
        "question": "How many continents are there on Earth?",
        "context": "Earth's land is divided into seven continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.",
        "answer": "Seven continents (Africa, Antarctica, Asia, Europe, North America, Oceania, and South America)",
        "reference_facts": ["There are 7 continents", "The continents are Africa, Antarctica, Asia, Europe, North America, Oceania, and South America"],
        "slice": ["factual", "easy", "geography"]
    },
    {
        "question": "What is photosynthesis?",
        "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water. It generally produces oxygen as a byproduct.",
        "answer": "Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to produce food and oxygen.",
        "reference_facts": ["Photosynthesis uses sunlight", "It converts CO2 and water into food", "It produces oxygen"],
        "slice": ["factual", "easy", "science"]
    },
]

FACTUAL_MEDIUM = [
    {
        "question": "What were the main causes of World War I?",
        "context": "World War I (1914-1918) was caused by several factors including: militarism and arms races, alliance systems that tied countries together, imperialism and competition for colonies, and nationalism. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria in Sarajevo on June 28, 1914.",
        "answer": "The main causes were militarism, alliance systems, imperialism, nationalism, and the immediate trigger was the assassination of Archduke Franz Ferdinand.",
        "reference_facts": ["Militarism and arms races", "Alliance systems", "Imperialism", "Nationalism", "Assassination of Franz Ferdinand was the trigger"],
        "slice": ["factual", "medium", "history"]
    },
    {
        "question": "How does the immune system fight viruses?",
        "context": "The immune system fights viruses through multiple mechanisms. First, innate immunity provides immediate defense through barriers and white blood cells. Then, adaptive immunity develops specific responses: B cells produce antibodies that neutralize viruses, while T cells destroy infected cells. Memory cells remember the virus for faster future responses.",
        "answer": "The immune system uses innate immunity (barriers, white blood cells) and adaptive immunity (B cells producing antibodies, T cells destroying infected cells, and memory cells for future protection).",
        "reference_facts": ["Innate immunity provides immediate defense", "B cells produce antibodies", "T cells destroy infected cells", "Memory cells enable faster future responses"],
        "slice": ["factual", "medium", "science"]
    },
]

REASONING_MEDIUM = [
    {
        "question": "If all roses are flowers and all flowers need water, do roses need water?",
        "context": "Logical reasoning problem involving categorical syllogisms.",
        "answer": "Yes, roses need water. Since all roses are flowers, and all flowers need water, roses must need water (transitive property).",
        "reference_facts": ["All roses are flowers", "All flowers need water", "Therefore roses need water"],
        "slice": ["reasoning", "medium", "logic"]
    },
    {
        "question": "A train leaves Station A at 60 mph. Another train leaves Station B (300 miles away) at 40 mph toward Station A. When will they meet?",
        "context": "Two trains traveling toward each other from stations 300 miles apart.",
        "answer": "They will meet in 3 hours. Combined speed is 100 mph (60+40), distance is 300 miles, so time = 300/100 = 3 hours.",
        "reference_facts": ["Train A: 60 mph", "Train B: 40 mph", "Distance: 300 miles", "Combined speed: 100 mph", "Time = Distance/Speed = 3 hours"],
        "slice": ["reasoning", "medium", "math"]
    },
]

MULTIHOP_HARD = [
    {
        "question": "Who directed the movie that won Best Picture at the Oscars in the year the iPhone was first released?",
        "context": "The iPhone was first released in 2007. At the 2008 Academy Awards (for 2007 films), 'No Country for Old Men' won Best Picture. The film was directed by Joel and Ethan Coen (the Coen Brothers).",
        "answer": "Joel and Ethan Coen (the Coen Brothers) directed 'No Country for Old Men', which won Best Picture for 2007 (the year iPhone was released).",
        "reference_facts": ["iPhone released in 2007", "No Country for Old Men won Best Picture for 2007", "Directed by Joel and Ethan Coen"],
        "slice": ["multi_hop", "hard", "entertainment"]
    },
    {
        "question": "What element has an atomic number equal to the number of strings on a standard guitar?",
        "context": "A standard guitar has 6 strings. The element with atomic number 6 is Carbon (C), which has 6 protons in its nucleus.",
        "answer": "Carbon, which has atomic number 6 (equal to the 6 strings on a standard guitar).",
        "reference_facts": ["Standard guitar has 6 strings", "Carbon has atomic number 6", "Atomic number = number of protons"],
        "slice": ["multi_hop", "hard", "science"]
    },
]


def generate_variations(templates, count, id_prefix):
    """Generate variations of template questions."""
    samples = []

    for i in range(count):
        template = random.choice(templates)
        sample = {
            "id": f"{id_prefix}_{i:04d}",
            **template
        }
        samples.append(sample)

    return samples


def generate_qa_dataset():
    """Generate complete Q&A dataset."""

    # Test set (120 samples - reduced from 150 to match JSON dataset size)
    test_samples = []
    test_samples.extend(generate_variations(FACTUAL_EASY, 40, "qa"))
    test_samples.extend(generate_variations(FACTUAL_MEDIUM, 30, "qa"))
    test_samples.extend(generate_variations(REASONING_MEDIUM, 30, "qa"))
    test_samples.extend(generate_variations(MULTIHOP_HARD, 20, "qa"))

    # Shuffle
    random.shuffle(test_samples)

    # Reassign sequential IDs
    for i, sample in enumerate(test_samples):
        sample["id"] = f"qa_{i:03d}"

    # Train set (2000 samples)
    train_samples = []
    train_samples.extend(generate_variations(FACTUAL_EASY, 700, "train_qa"))
    train_samples.extend(generate_variations(FACTUAL_MEDIUM, 600, "train_qa"))
    train_samples.extend(generate_variations(REASONING_MEDIUM, 500, "train_qa"))
    train_samples.extend(generate_variations(MULTIHOP_HARD, 200, "train_qa"))

    random.shuffle(train_samples)

    for i, sample in enumerate(train_samples):
        sample["id"] = f"train_qa_{i:04d}"

    # Validation set (30 samples)
    val_samples = []
    val_samples.extend(generate_variations(FACTUAL_EASY, 10, "val_qa"))
    val_samples.extend(generate_variations(FACTUAL_MEDIUM, 8, "val_qa"))
    val_samples.extend(generate_variations(REASONING_MEDIUM, 7, "val_qa"))
    val_samples.extend(generate_variations(MULTIHOP_HARD, 5, "val_qa"))

    random.shuffle(val_samples)

    for i, sample in enumerate(val_samples):
        sample["id"] = f"val_qa_{i:03d}"

    return train_samples, val_samples, test_samples


def save_dataset(samples, filepath):
    """Save samples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(samples)} samples to {filepath}")


def create_slices_yaml():
    """Create slices.yaml for Q&A task."""
    slices = {
        "slices": {
            "difficulty": {
                "easy": "Simple factual questions requiring single-fact retrieval",
                "medium": "Questions requiring understanding and synthesis",
                "hard": "Complex multi-hop reasoning questions"
            },
            "category": {
                "factual": "Direct fact retrieval",
                "reasoning": "Logical reasoning and inference",
                "multi_hop": "Requires connecting multiple pieces of information"
            },
            "domain": {
                "geography": "Geography and world knowledge",
                "science": "Scientific concepts and facts",
                "history": "Historical events and context",
                "math": "Mathematical reasoning",
                "logic": "Logical reasoning",
                "entertainment": "Movies, music, culture"
            }
        }
    }
    return slices


def main():
    """Generate and save Q&A datasets."""
    print("Generating Q&A datasets...")

    base_path = Path(__file__).parent.parent / "data" / "tasks" / "qa"

    # Generate datasets
    train_samples, val_samples, test_samples = generate_qa_dataset()

    # Save datasets
    save_dataset(train_samples, base_path / "train.jsonl")
    save_dataset(val_samples, base_path / "validation.jsonl")
    save_dataset(test_samples, base_path / "test.jsonl")

    # Save slices config
    import yaml
    slices = create_slices_yaml()
    with open(base_path / "slices.yaml", 'w') as f:
        yaml.dump(slices, f, default_flow_style=False)
    print(f"Saved slices config to {base_path / 'slices.yaml'}")

    print("\nDataset Summary:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)} samples")


if __name__ == "__main__":
    main()
