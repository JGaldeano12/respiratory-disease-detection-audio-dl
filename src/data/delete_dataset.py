import shutil, os

def delete_dataset():
    """
    Delete the training and testing dataset directories.

    The function removes the `Train` and `Test` directories located inside
    `/app/data/processed` if they exist. The parent `processed` directory and
    any other files or directories it contains are preserved.

    Returns:
        str: Confirmation message indicating that the dataset directories were
            deleted successfully.
    """
    # Delete the Train and Test directories if they exist
    path = '/app/data/processed'

    if os.path.exists(os.path.join(path, 'Train')):
        shutil.rmtree(os.path.join(path, 'Train'))
    
    if os.path.exists(os.path.join(path, 'Test')):
        shutil.rmtree(os.path.join(path, 'Test'))

    return "Dataset deleted successfully."