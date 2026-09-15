import argparse
from methods_in_ai_research.processing import preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to the data file")
    args = parser.parse_args()
    
    data = preprocess(args.data_path)

    
if __name__ == "__main__":
    main()
