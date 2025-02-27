import os
import pathlib
import pickle
import pandas as pd


def convert_pkl_to_csv(input_folder, output_folder):
    # List all .pkl files in the input folder
    pkl_filepaths = sorted(pathlib.Path(input_folder).rglob('*.pkl'))

    if len(pkl_filepaths) == 0:
        print("No .pkl files found in the folder.")
        return

    # Process each .pkl file
    for pkl_filepath in pkl_filepaths:
        csv_file_path = os.path.join(
            output_folder,
            pkl_filepath.with_suffix(".pcd").relative_to(input_folder)
        )
        try:
            # Load the .pkl file
            with open(pkl_filepath, 'rb') as file:
                data = pickle.load(file)

            # Convert data to DataFrame if needed
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame(data)

            # Save to .csv file
            df.to_csv(csv_file_path, index=False)
            print(f"Successfully converted '{pkl_filepath}' to '{csv_file_path}'")

        except Exception as e:
            print(f"Failed to process {pkl_filepath}: {e}")


def test_convert_pkl_to_csv():
    input_folder = r"pandaset/001/annotations/cuboids"
    output_folder = r"pandaset/001/annotations/cuboids"
    convert_pkl_to_csv(input_folder=input_folder, output_folder=output_folder)


if __name__ == "__main__":
    test_convert_pkl_to_csv()
