import scipy.io as io

def load_mat_file(file_path):
    try:
        data = io.loadmat(file_path)
        print("File loaded successfully.")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_file.mat' with the path to your .mat file
mat_file_path = 'attention_behaviorals.mat'
data = load_mat_file(mat_file_path)

# You can now work with 'data', which is a dictionary containing the variables from the .mat file
def inspect_mat_data(data):
    print("Inspecting .mat file contents...")

    # Print the type of the data structure
    print(f"Type of data: {type(data)}")

    # Iterate over all items in the dictionary and print their keys and types
    for key, value in data.items():
        print(f"Key: '{key}' - Type: {type(value)}")

        # If the value is a numpy array, you can also print its shape
        if hasattr(value, 'shape'):
            print(f"Shape of '{key}': {value.shape}")

        # Optionally, print a small part of the value if it's large
        if isinstance(value, (list, dict)):
            print(f"Contents of '{key}': {str(value)[:100]}...")

# Load the .mat file and inspect its contents
inspect_mat_data(data)
