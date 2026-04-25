from service.synthetic_data import generate_synthetic_data


if __name__ == "__main__":
    result = generate_synthetic_data()
    print(
        "Synthetic data created: "
        f"{result['meta_rows']} meta rows, {result['usage_rows']} usage rows."
    )
