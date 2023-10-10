import os
import random
import shutil

# Rock paper scissors step by step  with tensorflow
#
# build an image recogniton model


# Define a function to split data into train and val sets
def split_data(data_path, output_path, train_ratio=0.85, val_ratio=0.15):
    # Create output folders for train and val
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists or create it

    # declare path for training data folder and the validation data folder where we will move the images to

    train_folder = os.path.join(output_path, 'train')  # declare the  Path for the  training data folder
    val_folder = os.path.join(output_path, 'val')      # Path for the validation data folder

    print(train_folder)

    # now create the training and validation folder where we will move the images to

    os.makedirs(train_folder, exist_ok=True)  # Create the training data folder if it doesn't exist
    os.makedirs(val_folder, exist_ok=True)    # Create the validation data folder if it doesn't exist


    # create a list of all the data categories we have on the present dataset folder

    categories = ["paper", "rock", "scissors"]  # List of categories in your dataset


    # next create a loop where we will go through all the images on the data set and copy it

    for category in categories:

        # we will print the image categories first

        print(category)

        # next we will write a code to join this with our data path

       # write a code to join this categories and our dataset folder so we can get to it
       #  which is the path to the Path to the category folder in the source data

        category_folder = os.path.join(data_path, category)  # Path to the category folders in the source data

        print("ca",category_folder)
        # check for image extensions
        # # Create a list of image files with valid extensions

        valid_image_files = []

        for filename in os.listdir(category_folder): #list all the files in the dataset filders

            # print(filename)

            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):


                # print(valid_image_files)

                valid_image_files.append(filename)

        # # Shuffle the list of valid image files randomly
        random.shuffle(valid_image_files)

        # # Calculate the total number of valid images in the category

        num_images = len(valid_image_files) #check the numbers of valid images we have

        # # Calculate the number or pecentage  of images for training and validation
        #this will be used to split  our images to their right folders

        num_train = int(num_images * train_ratio)
        print(num_train)

        num_val = int(num_images * val_ratio)
        print(num_val)

        # next we will Create subfolders for each category in both train and val folders for our new folder
        print(train_folder, category)

        #firstly declear the path to the train_category_folder and val _category_folder for the new folder
        train_category_folder = os.path.join(train_folder, category)  # declear Path for the category folders in the training data
        val_category_folder   = os.path.join(val_folder, category)      # Path for the category in the validation data


        #make or create the folder

        os.makedirs(train_category_folder, exist_ok=True)  # Create the category folder in training if it doesn't exist
        os.makedirs(val_category_folder, exist_ok=True)    # Create the category folder in validation if it doesn't exist

#select the images we will move to the new folder the num of images will be based on what we gave before
        print(num_train)
        print(num_val)
        train_data = valid_image_files[:num_train]  #starting from 0 to the amount of image num, Select the first num_train images for training

        # Select the next num_val images for validation ,starting from where we stoped (num_train) in train
        # to the end/last of images (num_train+num_val)

        val_data = valid_image_files[num_train:num_train+num_val]  # Select the next num_val images for validation

        # print(data)
        # print(val_data)


        # copy the image

        #loop through the tain data set  and copy it
        for filename in train_data:

            print(filename)

            #declear the  Source path for the image in the category folder
            #hope you remember the category folder which is our dataset categori which we got in 41
            # we will join this to the image names  and copy


        # so we join this main data set path  with the filename we got so we can locate it and copy
            src_path = os.path.join(category_folder, filename)  # Source path for the image in the category folder
            print(category_folder, filename)

            #declear the destination path
            dest_path = os.path.join(train_category_folder, filename)  # Destination path for the image in the training category folder
            # coopy
            shutil.copy(src_path, dest_path)  # Copy the image from source to destinatio

        #do came for val

        for filename in val_data:
            src_path = os.path.join(category_folder, filename)  # Source path for the image in the category folder
            dest_path = os.path.join(val_category_folder, filename)  # Destination path for the image in the validation category folder
            shutil.copy(src_path, dest_path)  # Copy the image from source to destination
        #
        #

# Entry point of the script
if __name__ == '__main__':
    data_path = "dataset"  # Path to the source dataset folder
    output_path = "data"  # Path where the split data will be saved
    split_data(data_path, output_path)  # Call the split_data function to perform the data splitting


    print(len(os.listdir("data/train/paper")))
    print(len(os.listdir("data/train/rock")))
    print(len(os.listdir("data/train/scissors")))

    print(len(os.listdir("data/val/paper")))
    print(len(os.listdir("data/val/rock")))
    print(len(os.listdir("data/val/scissors")))

    print("--------------------------------------------------")

    print(len(os.listdir("dataset/paper")))
    print(len(os.listdir("dataset/rock")))
    print(len(os.listdir("dataset/scissors")))
    print("--------------------------------------------------")
    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/train/paper")))
    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/train/rock")))
    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/train/scissors")))

    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/validation/paper")))
    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/validation/rock")))
    print(len(os.listdir("Rock-Paper-Scissors-image-recognition/validation/scissors")))


