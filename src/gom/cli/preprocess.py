"""Installed CLI entry point for image preprocessing."""

def main():
    """Main entry point for the ``gom-preprocess`` command."""
    from image_preprocessor import main as preprocess_main
    return preprocess_main()

if __name__ == "__main__":
    main()
