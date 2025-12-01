"""
Generate synthetic datasets for JSON extraction and Q&A tasks.

This script creates:
- 150 test cases for JSON extraction (80% test, 20% validation)
- 150 test cases for Q&A (80% test, 20% validation)
- 2000-3000 training samples for each task (for finetuning)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Set seed for reproducibility
random.seed(42)

def generate_json_extraction_cases() -> List[Dict[str, Any]]:
    """Generate JSON extraction test cases with varying difficulty."""

    test_cases = []
    case_id = 1

    # Easy cases (50): Simple flat objects
    easy_templates = [
        {
            "input": "Name: {name}, Age: {age}, City: {city}",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"}
                },
                "required": ["name", "age", "city"]
            }
        },
        {
            "input": "Product: {product}, Price: ${price}, Stock: {stock} units",
            "schema": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"},
                    "price": {"type": "number"},
                    "stock": {"type": "integer"}
                },
                "required": ["product", "price", "stock"]
            }
        },
        {
            "input": "Email: {email}, Phone: {phone}, Active: {active}",
            "schema": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "active": {"type": "boolean"}
                },
                "required": ["email", "phone", "active"]
            }
        }
    ]

    # Sample data
    names = ["Alice Johnson", "Bob Smith", "Carol Davis", "David Brown", "Eve Wilson"]
    cities = ["New York", "San Francisco", "Austin", "Seattle", "Boston"]
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"]
    emails = [f"{name.split()[0].lower()}@example.com" for name in names]
    phones = [f"+1-555-{random.randint(1000,9999)}" for _ in range(5)]

    for i in range(50):
        template = random.choice(easy_templates)

        # Fill template
        if "name" in template["input"]:
            data = {
                "name": random.choice(names),
                "age": random.randint(25, 65),
                "city": random.choice(cities)
            }
        elif "product" in template["input"]:
            data = {
                "product": random.choice(products),
                "price": round(random.uniform(10, 1000), 2),
                "stock": random.randint(0, 100)
            }
        else:
            data = {
                "email": random.choice(emails),
                "phone": random.choice(phones),
                "active": random.choice([True, False])
            }

        input_text = template["input"].format(**data)

        test_cases.append({
            "id": f"json_{case_id:03d}",
            "input": input_text,
            "expected_output": data,
            "schema": template["schema"],
            "slice": ["easy", "flat_object"],
            "difficulty": "easy"
        })
        case_id += 1

    # Medium cases (50): Nested objects and arrays
    for i in range(50):
        # Person with address
        person = {
            "name": random.choice(names),
            "age": random.randint(25, 65),
            "address": {
                "street": f"{random.randint(100,999)} Main St",
                "city": random.choice(cities),
                "zip": f"{random.randint(10000,99999)}"
            },
            "skills": random.sample(["Python", "JavaScript", "Go", "Rust", "Java"], k=3)
        }

        input_text = f"""
        Person Information:
        Name: {person['name']}
        Age: {person['age']}
        Address: {person['address']['street']}, {person['address']['city']}, {person['address']['zip']}
        Skills: {', '.join(person['skills'])}
        """

        test_cases.append({
            "id": f"json_{case_id:03d}",
            "input": input_text.strip(),
            "expected_output": person,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                            "zip": {"type": "string"}
                        }
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "slice": ["medium", "nested_object", "array"],
            "difficulty": "medium"
        })
        case_id += 1

    # Hard cases (30): Complex nested structures
    for i in range(30):
        company = {
            "name": f"{random.choice(['Tech', 'Data', 'Cloud', 'AI'])} Corp",
            "founded": random.randint(2000, 2020),
            "employees": [
                {
                    "id": j,
                    "name": random.choice(names),
                    "role": random.choice(["Engineer", "Manager", "Designer"]),
                    "salary": random.randint(80000, 200000)
                }
                for j in range(1, 4)
            ],
            "revenue": {
                "2022": round(random.uniform(1, 100), 2),
                "2023": round(random.uniform(1, 100), 2)
            }
        }

        input_text = f"""
        Company: {company['name']}
        Founded: {company['founded']}
        Employees:
        """
        for emp in company['employees']:
            input_text += f"\n  - ID: {emp['id']}, Name: {emp['name']}, Role: {emp['role']}, Salary: ${emp['salary']}"

        input_text += f"\nRevenue: 2022: ${company['revenue']['2022']}M, 2023: ${company['revenue']['2023']}M"

        test_cases.append({
            "id": f"json_{case_id:03d}",
            "input": input_text.strip(),
            "expected_output": company,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "founded": {"type": "integer"},
                    "employees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "role": {"type": "string"},
                                "salary": {"type": "integer"}
                            }
                        }
                    },
                    "revenue": {"type": "object"}
                }
            },
            "slice": ["hard", "nested_object", "array_of_objects"],
            "difficulty": "hard"
        })
        case_id += 1

    # Domain-specific cases (20): Technical/specialized
    for i in range(20):
        api_response = {
            "status": random.choice([200, 201, 400, 500]),
            "data": {
                "user_id": random.randint(1000, 9999),
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {
                    "ip": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                    "user_agent": "Mozilla/5.0"
                }
            },
            "errors": [] if random.random() > 0.3 else ["Invalid token"]
        }

        input_text = f"""
        API Response:
        Status Code: {api_response['status']}
        User ID: {api_response['data']['user_id']}
        Timestamp: {api_response['data']['timestamp']}
        IP Address: {api_response['data']['metadata']['ip']}
        User Agent: {api_response['data']['metadata']['user_agent']}
        Errors: {', '.join(api_response['errors']) if api_response['errors'] else 'None'}
        """

        test_cases.append({
            "id": f"json_{case_id:03d}",
            "input": input_text.strip(),
            "expected_output": api_response,
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "integer"},
                    "data": {"type": "object"},
                    "errors": {"type": "array", "items": {"type": "string"}}
                }
            },
            "slice": ["domain_specific", "api_response"],
            "difficulty": "medium"
        })
        case_id += 1

    return test_cases


def generate_qa_cases() -> List[Dict[str, Any]]:
    """Generate Q&A test cases with varying difficulty."""

    test_cases = []
    case_id = 1

    # Factual questions (60): Direct fact retrieval
    factual_qa = [
        {
            "context": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant indentation.",
            "question": "Who created Python?",
            "answer": "Guido van Rossum",
            "facts": ["Guido van Rossum created Python"]
        },
        {
            "context": "The Earth has a radius of approximately 6,371 kilometers at the equator. It takes 365.25 days to orbit the Sun, which is why we have leap years every four years.",
            "question": "How long does Earth take to orbit the Sun?",
            "answer": "365.25 days",
            "facts": ["Earth takes 365.25 days to orbit the Sun"]
        },
        {
            "context": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and learn from it.",
            "question": "What is machine learning?",
            "answer": "A subset of artificial intelligence that enables systems to learn from experience without explicit programming",
            "facts": ["Machine learning is a subset of AI", "Systems learn from experience"]
        }
    ]

    # Duplicate and vary the factual questions
    for i in range(60):
        qa = random.choice(factual_qa)
        test_cases.append({
            "id": f"qa_{case_id:03d}",
            "question": qa["question"],
            "context": qa["context"],
            "expected_answer": qa["answer"],
            "reference_facts": qa["facts"],
            "slice": ["factual", "single_hop"],
            "difficulty": "easy"
        })
        case_id += 1

    # Reasoning questions (50): Require inference
    reasoning_qa = [
        {
            "context": "The company's revenue was $10M in 2022 and $15M in 2023. Operating costs were $8M in 2022 and $11M in 2023.",
            "question": "Did the company's profit increase or decrease from 2022 to 2023?",
            "answer": "The profit increased from $2M ($10M - $8M) in 2022 to $4M ($15M - $11M) in 2023",
            "facts": ["Revenue and costs for both years", "Profit = Revenue - Costs"]
        },
        {
            "context": "All birds have feathers and lay eggs. Penguins are birds that cannot fly but are excellent swimmers.",
            "question": "Do penguins lay eggs?",
            "answer": "Yes, penguins lay eggs because they are birds and all birds lay eggs",
            "facts": ["All birds lay eggs", "Penguins are birds"]
        }
    ]

    for i in range(50):
        qa = random.choice(reasoning_qa)
        test_cases.append({
            "id": f"qa_{case_id:03d}",
            "question": qa["question"],
            "context": qa["context"],
            "expected_answer": qa["answer"],
            "reference_facts": qa["facts"],
            "slice": ["reasoning", "inference"],
            "difficulty": "medium"
        })
        case_id += 1

    # Multi-hop questions (40): Require multiple steps
    multihop_qa = [
        {
            "context": "Alice manages the engineering team. The engineering team reports to Bob, who is the CTO. The CTO reports directly to the CEO, Carol.",
            "question": "Who is Alice's boss's boss?",
            "answer": "Carol, the CEO, because Alice reports to Bob (CTO), and Bob reports to Carol",
            "facts": ["Alice's boss is Bob", "Bob's boss is Carol"]
        }
    ]

    for i in range(40):
        qa = random.choice(multihop_qa)
        test_cases.append({
            "id": f"qa_{case_id:03d}",
            "question": qa["question"],
            "context": qa["context"],
            "expected_answer": qa["answer"],
            "reference_facts": qa["facts"],
            "slice": ["multi_hop", "reasoning"],
            "difficulty": "hard"
        })
        case_id += 1

    return test_cases


def generate_training_data(task_type: str, num_samples: int = 2000) -> List[Dict[str, Any]]:
    """Generate training data for finetuning."""

    if task_type == "json":
        # Generate more JSON extraction examples
        training_data = []
        for i in range(num_samples):
            # Simpler variations for training
            case = generate_json_extraction_cases()[0]  # Get one template
            case["id"] = f"train_json_{i:04d}"
            training_data.append(case)
        return training_data

    elif task_type == "qa":
        # Generate more Q&A examples
        training_data = []
        for i in range(num_samples):
            case = generate_qa_cases()[0]
            case["id"] = f"train_qa_{i:04d}"
            training_data.append(case)
        return training_data

    return []


def main():
    """Generate all datasets."""

    print("🚀 Generating datasets for LLM evaluation harness...")

    # Create output directories
    json_dir = Path("data/tasks/json_extraction")
    qa_dir = Path("data/tasks/qa")
    json_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Generate JSON extraction datasets
    print("\n📊 Generating JSON extraction datasets...")
    json_cases = generate_json_extraction_cases()
    print(f"  ✓ Generated {len(json_cases)} JSON extraction test cases")

    # Split into test and validation
    random.shuffle(json_cases)
    json_test = json_cases[:120]  # 80%
    json_val = json_cases[120:]   # 20%

    # Save test set
    with open(json_dir / "test.jsonl", "w") as f:
        for case in json_test:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(json_test)} test cases to test.jsonl")

    # Save validation set
    with open(json_dir / "validation.jsonl", "w") as f:
        for case in json_val:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(json_val)} validation cases to validation.jsonl")

    # Generate and save training data
    print("\n🔄 Generating training data for JSON extraction...")
    json_train = generate_training_data("json", num_samples=2000)
    with open(json_dir / "train.jsonl", "w") as f:
        for case in json_train:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(json_train)} training cases to train.jsonl")

    # Create slices configuration
    json_slices = {
        "slices": {
            "easy": [c["id"] for c in json_cases if "easy" in c.get("slice", [])],
            "medium": [c["id"] for c in json_cases if "medium" in c.get("slice", [])],
            "hard": [c["id"] for c in json_cases if "hard" in c.get("slice", [])],
            "domain_specific": [c["id"] for c in json_cases if "domain_specific" in c.get("slice", [])]
        }
    }
    with open(json_dir / "slices.yaml", "w") as f:
        yaml.dump(json_slices, f, default_flow_style=False)
    print(f"  ✓ Created slices configuration")

    # Generate Q&A datasets
    print("\n💬 Generating Q&A datasets...")
    qa_cases = generate_qa_cases()
    print(f"  ✓ Generated {len(qa_cases)} Q&A test cases")

    # Split into test and validation
    random.shuffle(qa_cases)
    qa_test = qa_cases[:120]  # 80%
    qa_val = qa_cases[120:]   # 20%

    # Save test set
    with open(qa_dir / "test.jsonl", "w") as f:
        for case in qa_test:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(qa_test)} test cases to test.jsonl")

    # Save validation set
    with open(qa_dir / "validation.jsonl", "w") as f:
        for case in qa_val:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(qa_val)} validation cases to validation.jsonl")

    # Generate and save training data
    print("\n🔄 Generating training data for Q&A...")
    qa_train = generate_training_data("qa", num_samples=2000)
    with open(qa_dir / "train.jsonl", "w") as f:
        for case in qa_train:
            f.write(json.dumps(case) + "\n")
    print(f"  ✓ Saved {len(qa_train)} training cases to train.jsonl")

    # Create slices configuration
    qa_slices = {
        "slices": {
            "factual": [c["id"] for c in qa_cases if "factual" in c.get("slice", [])],
            "reasoning": [c["id"] for c in qa_cases if "reasoning" in c.get("slice", [])],
            "multi_hop": [c["id"] for c in qa_cases if "multi_hop" in c.get("slice", [])]
        }
    }
    with open(qa_dir / "slices.yaml", "w") as f:
        yaml.dump(qa_slices, f, default_flow_style=False)
    print(f"  ✓ Created slices configuration")

    # Print summary
    print("\n✅ Dataset generation complete!")
    print(f"\n📈 Summary:")
    print(f"  JSON Extraction: {len(json_test)} test + {len(json_val)} val + {len(json_train)} train")
    print(f"  Q&A: {len(qa_test)} test + {len(qa_val)} val + {len(qa_train)} train")
    print(f"\n📁 Datasets saved to:")
    print(f"  - {json_dir}")
    print(f"  - {qa_dir}")


if __name__ == "__main__":
    main()
