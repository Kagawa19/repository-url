# Expert Resource Matcher

This project connects experts to their resources based on matching author names in a database. It provides functionality to load expert data, normalize names, and link experts to resources efficiently.

## Project Structure

```
expert-resource-matcher
├── src
│   ├── database
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   └── models.py
│   ├── matching
│   │   ├── __init__.py
│   │   ├── matcher.py
│   │   └── normalizer.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── logger.py
│   └── config.py
├── tests
│   ├── __init__.py
│   ├── test_matcher.py
│   └── test_normalizer.py
├── requirements.txt
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Set up your database and configure the connection parameters in `src/config.py`.
2. Load expert data into the database.
3. Use the `Matcher` class from `src/matching/matcher.py` to link experts to resources based on author names.

## Testing

To run the tests, use:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.