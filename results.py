from scripts.results import *
from scripts.test import *
from scripts.utils import *

if __name__ == "__main__":
    # display_models_summary("output/train")
    # display_models_barplots_multiple("output/test")
    # test_taxonomic_loss()
    classes_to_remove = []
    create_dataset_test(classes_to_remove)