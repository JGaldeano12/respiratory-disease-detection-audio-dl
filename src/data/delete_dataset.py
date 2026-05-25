import shutil, os

def delete_dataset():
    # I want to delete only the Train and Test folders within the processed directory, but not the processed directory itself, since it may contain other files or folders that I want to keep.
    path = '/app/data/processed'

    if os.path.exists(os.path.join(path, 'Train')):
        shutil.rmtree(os.path.join(path, 'Train'))
    
    if os.path.exists(os.path.join(path, 'Test')):
        shutil.rmtree(os.path.join(path, 'Test'))

    return "Dataset deleted successfully."

delete_dataset()