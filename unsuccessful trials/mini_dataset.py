from shutil import copytree, rmtree
import os


# Helper method to create train_mini and test_mini data samples
def dataset_mini(food_list, src, dest):
    if os.path.exists(dest):
        rmtree(
            dest)  # removing dataset_mini(if it already exists) folders so that we will have only the classes that we
        # want
    os.makedirs(dest)
    for food_item in food_list:
        print("Copying images into", food_item)
        copytree(os.path.join(src, food_item), os.path.join(dest, food_item))


# picking 3 food items and generating separate data folders for the same
food_list = ['falafel', 'pizza', 'omelette', 'hamburger', 'sushi']
src_train = 'train'
dest_train = 'train_mini'
src_test = 'test'
dest_test = 'test_mini'

print("Creating train data folder with new classes")
dataset_mini(food_list, 'food-101/train', 'food-101/mini_train')

print("Creating test data folder with new classes")
dataset_mini(food_list, 'food-101/test', 'food-101/mini_test')
