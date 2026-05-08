import kagglehub
import os
DOWNLOAD_PATH = "./data/semi-aves"
SYMBOLIC_LINK_PATH = "./src/dataset"

# Download data
try: 
    os.mkdir(DOWNLOAD_PATH)
except (FileExistsError, FileNotFoundError) as err:
    print(f"{DOWNLOAD_PATH} Aves Download Path Already exists")

try: 
    os.mkdir(SYMBOLIC_LINK_PATH)
except (FileExistsError, FileNotFoundError) as err:
    print(f"{SYMBOLIC_LINK_PATH} Aves Symbolic Link Path Already exists")

# Prompt for user API key
kagglehub.login()
# Download latest version
path = kagglehub.competition_download('semi-inat-2020', output_dir = DOWNLOAD_PATH)

print("Path to Aves files:", path)

# Create Symbolic Links
symlink_base_path = os.path.join(SYMBOLIC_LINK_PATH, 'semi-aves')

# Annotation
symlink_annotation_path = os.path.join(SYMBOLIC_LINK_PATH, 'annotation')
real_path_annotation = os.path.join(path, "annotation", "annotation")

try:
    os.symlink(real_path_annotation, symlink_annotation_path)
    print(f'Created symbolic link: {symlink_annotation_path} -> {real_path_annotation}')
except OSError as e:
    print(f'Error creating symbolic link: {e}')

# Test
symlink_test_path = os.path.join(SYMBOLIC_LINK_PATH, 'test')
real_path_test = os.path.join(path, "test", "test")

try:
    os.symlink(real_path_test, symlink_test_path)
    print(f'Created symbolic link: {symlink_test_path} -> {real_path_test}')
except OSError as e:
    print(f'Error creating symbolic link: {e}')

# Trainval images 
symlink_trainval_images_path = os.path.join(SYMBOLIC_LINK_PATH, 'trainval_images')
real_path_trainval_images = os.path.join(path, "trainval_images", "trainval_images")

try:
    os.symlink(real_path_trainval_images, symlink_trainval_images_path)
    print(f'Created symbolic link: {symlink_trainval_images_path} -> {real_path_trainval_images}')
except OSError as e:
    print(f'Error creating symbolic link: {e}')

# U train in
symlink_trainval_u_train_in_path = os.path.join(SYMBOLIC_LINK_PATH, 'u_train_in')
real_path_trainval_u_train_in_path = os.path.join(path, "u_train_in", "u_train_in")

try:
    os.symlink(real_path_trainval_u_train_in_path, symlink_trainval_u_train_in_path)
    print(f'Created symbolic link: {symlink_trainval_u_train_in_path} -> {real_path_trainval_u_train_in_path}')
except OSError as e:
    print(f'Error creating symbolic link: {e}')

# U train out
symlink_trainval_u_train_out_path = os.path.join(SYMBOLIC_LINK_PATH, 'u_train_out')
real_path_trainval_u_train_out_path = os.path.join(path, "u_train_out", "u_train_out")

try:
    os.symlink(real_path_trainval_u_train_out_path, symlink_trainval_u_train_out_path)
    print(f'Created symbolic link: {symlink_trainval_u_train_out_path} -> {real_path_trainval_u_train_out_path}')
except OSError as e:
    print(f'Error creating symbolic link: {e}')

print("Symbolic links created successfully")