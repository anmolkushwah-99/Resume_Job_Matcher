"""
Environment and Dependency Verification Script
Phase 1: Project Foundation and Development Environment Setup
"""

import sys
import importlib
import importlib.metadata

def verify_environment():
    # 1. Verify Core Packages
    packages = [
        ("pandas", "Pandas", "pandas"),
        ("numpy", "NumPy", "numpy"),
        ("sklearn", "Scikit-learn", "scikit-learn"),
        ("nltk", "NLTK", "nltk"),
        ("spacy", "spaCy", "spacy"),
        ("flask", "Flask", "flask"),
        ("pypdf", "pypdf", "pypdf"),
        ("flask_cors", "flask-cors", "flask-cors"),
        ("matplotlib", "Matplotlib", "matplotlib"),
        ("seaborn", "Seaborn", "seaborn"),
    ]
    
    missing_packages = []
    installed_versions = {}
    
    for mod_name, display_name, dist_name in packages:
        try:
            importlib.import_module(mod_name)
            try:
                ver = importlib.metadata.version(dist_name)
            except Exception:
                mod = sys.modules.get(mod_name)
                ver = getattr(mod, "__version__", "Installed")
            installed_versions[display_name] = ver
        except ImportError as e:
            missing_packages.append((display_name, str(e)))

    if missing_packages:
        print("[-] Verification Failed: Missing required packages:")
        for name, err in missing_packages:
            print(f"    - {name}: {err}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)

    # 2. Verify spaCy English Model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"[-] Verification Failed: spaCy model 'en_core_web_sm' could not be loaded: {e}")
        print("Please run: python -m spacy download en_core_web_sm")
        sys.exit(1)

    # 3. Verify NLTK Resources
    try:
        import nltk
        from nltk.corpus import stopwords, wordnet
        from nltk.tokenize import word_tokenize

        # Test tokenization
        _ = word_tokenize("Testing NLTK punkt tokenization.")
        # Test stopwords
        _ = stopwords.words("english")
        # Test wordnet
        _ = wordnet.synsets("test")
    except Exception as e:
        print(f"[-] Verification Failed: Error verifying NLTK resources: {e}")
        print("Please run: python -m nltk.downloader punkt stopwords wordnet punkt_tab")
        sys.exit(1)

    # 4. Print Success Summary
    print("=" * 55)
    print("Environment setup successful!")
    print("=" * 55)
    print(f"Python:       {sys.version.split()[0]}")
    for name, ver in installed_versions.items():
        print(f"{name + ':':<15} {ver}")
    print(f"{'spaCy model:':<15} en_core_web_sm (Loaded successfully)")
    print(f"{'NLTK data:':<15} punkt, stopwords, wordnet (Verified)")
    print("=" * 55)

if __name__ == "__main__":
    verify_environment()
