import os
import shutil

def delete_dataset():
    """
    Delete the training, validation and testing dataset directories.

    The function removes the `Train`, `Val` and `Test` directories located
    inside `/app/data/processed` if they exist. The parent `processed`
    directory and any other files or directories it contains are preserved.

    Returns:
        str: Confirmation message indicating that the dataset directories were
            deleted successfully.
    """
    path = '/app/data/processed'

    # Delete the Train, Val and Test directories if they exist.
    for split in ['Train', 'Val', 'Test']:
        if os.path.exists(os.path.join(path, split)):
            shutil.rmtree(os.path.join(path, split))

    return "Dataset deleted successfully."