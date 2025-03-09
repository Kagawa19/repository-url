class Normalizer:
    """A class to normalize names for comparison."""

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize a name by stripping whitespace and converting to lowercase."""
        return ' '.join(name.lower().strip().split())

    @staticmethod
    def get_name_parts(author_name: str) -> tuple:
        """Extract first and last name from author string."""
        parts = author_name.strip().split()
        if len(parts) == 1:
            return ('', parts[0])
        return (' '.join(parts[:-1]), parts[-1])