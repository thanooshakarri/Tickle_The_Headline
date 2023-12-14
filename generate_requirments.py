import pkg_resources

# List of libraries you mentioned
libraries = [
    "scikit-learn",
    "pandas",
    "numpy",
    "transformers",
    "datasets",
    "tensorflow",
]

# Create a requirements.txt file with library names and versions
with open("requirements.txt", "w") as f:
    for lib in libraries:
        try:
            # Get the version of the library
            version = pkg_resources.get_distribution(lib).version
            f.write(f"{lib}=={version}\n")
        except pkg_resources.DistributionNotFound:
            # Handle the case where the library is not installed
            pass
