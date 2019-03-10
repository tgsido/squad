import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file_name", default=None, type=str, required=True,
                        help="Filename for json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="Filename for output csv")
    parser.add_argument("--data_type", default=None, type=str, required=True,
                        help="specify test or dev")
    args = parser.parse_args()

    df = pd.read_json(args.input_file_name)
    df.to_csv(args.output_dir + "/" + args.data_type + "_submission.csv" )

if __name__ == "__main__":
    main()
