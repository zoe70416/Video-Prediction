# dlFinalProj


```markdown
# UNet for Multi-Class Semantic Segmentation

The UNet code is adapted from Mohammad Hamdaan's work on "Multi-Class Semantic Segmentation with U-Net & PyTorch" ([Medium Article](https://medium.com/@mhamdaan/multi-class-semantic-segmentation-with-u-net-pytorch-ee81a66bba89)).

The model structure can be found in the [model.py](https://github.com/hamdaan19/UNet-Multiclass/blob/main/scripts/model.py) file.

## Dataset Structure

Before running the UNet code, ensure the dataset is preprocessed and follows this structure:


Dataset_Student
  - train
       - img
       - mask
  - val
      - img
      - mask
```

Images are named in the format: `video_{video number}_image_{frame_number}.png`, and masks: `video_{video number}_mask_{frame_number}.png`.

## Create conda environment

Create a conda environment for running the UNet code by using the requirements.txt file

```python
conda create --name your_environment_name --file requirements.txt
```

## Data Preparation

To prepare the data, follow these commands:

a. Use `data_prep.py` to prepare mask data for train and val:

```bash
# For train - mask
python3 data_prep.py dir_of_where_original_folder_is_saved dir_where_you_want_to_save_your_mask

# For val - mask
python3 data_prep.py dir_of_where_original_folder_is_saved dir_where_you_want_to_save_your_mask
```

Example for training data:

```python
python3 /home/ki2130/dlFinalProj/data_prep.py /scratch/ki2130/train/mask
```

b. Use `rename.py` to prepare image data for train and val:

```bash
# For train - img
python3 rename.py dir_of_where_original_folder_is_saved dir_where_you_want_to_save_your_image

# For val - img
python3 rename.py dir_of_where_original_folder_is_saved dir_where_you_want_to_save_your_image
```
Example for training data:

```python
python3 /home/ki2130/dlFinalProj/rename.py /scratch/ki2130/train/img
```

## Unet Utils *OPTIONAL*

- In *unet_utils.py*

- If you would like to output some graphs of the Training Loss and the Jaccard Index during training edit *unet_utils.py* as follows:

- Change the filepath to the training_loss_plot to a path of your choice:

```python
  plot_path = "/mnt/home/kinchoco/ceph/dl-project/mask_metrics/training_loss_plot.png"
```
- Change the filepath to the training_jaccard_plot to a path of your choice:

```python
plot_path = "/mnt/home/kinchoco/ceph/dl-project/mask_metrics/training_jaccard_plot.png"
```
## Unet Train

## IMPORTANT FOLLOW THE STEPS CAREFULLY 

- In *unet_train.py*

- Change the MODEL_PATH to where you would like *unet_multi_class_segmentation.pth* to live. In the example below, we save the UNet model output to scratch in GCP. This will save an intermediate UNet model during training after every batch.

```python
MODEL_PATH = "/scratch/ki2130/mask_models/unet_multi_class_segmentation.pth"
```

- SET LOAD_MODEL = False 

- In the original implementation (where we ran on the Flatiron cluster), we set BATCH_SIZE = 256 and NUM_WORKERS = 22. In order to run on 1 GPU node in the NYU GPC use:

```python
BATCH_SIZE = 16
NUM_WORKERS = 8
```

- Set train_root_dir to where the "train" folder lives from the Data Preparation step. This folder should contain the "img" and "mask" subfolders. DO NOT use the original training dataset from the raw data.

For example:
```python
train_root_dir = '/scratch/ki2130/train'
```

- Change the output of the final trained UNet model which will populate only AFTER all epochs have COMPLETED.

For example:
```python
torch.save(unet.state_dict(), "/scratch/ki2130/mask_models/unet_multi_class_segmentation_v1.pth")
```

## Reproduce the results:

- Step 1: Run the script using this command

```python
python3 file_path_to_unet_train.py --epochs 8
```

For example:

```python
python3  /home/ki2130/dlFinalProj/unet_train.py --epochs 8
```

- Step 2: Continue training. Run the script using this command but ONLY on a smaller version of the "train" folder from Data Preparation that has only the first 2 videos from training. We included this dataset with the code it is called "train_mini". Make sure to change the train_root_dir, SET LOAD = True, and use the filepath of the trained model in Step 1 as input to the MODEL_PATH:

```python
python3 file_path_to_unet_train.py --epochs 90
```

For example:

```python
python3  /home/ki2130/dlFinalProj/unet_train.py --epochs 90
```

For example:
```python
MODEL_PATH = file_path_to_trained_model_from_Step_1
LOAD = True
train_root_dir = "/train_mini"
```

## Unet Pred

- In *unet_pred.py*

- Change the MODEL_PATH to where *unet_multi_class_segmentation_v6.pth* lives. We include this in the files. This is the final trained UNet model used to generate the predictions for the final leaderboard.

- Make sure LOAD_MODEL = True

- Change to the filepath to where you would like the final output array to be saved

```python
MODEL_PATH = "/scratch/ki2130/unet_multi_class_segmentation_v6.pth" 
save_output_array = '/scratch/ki2130/mask_preds/output_array.npy'
```
- RUN the following command and make sure after --dir to include the filepath to where "hidden-practice2-sun5pm" lives. These are the frame predictions from the PredNet Model that we pass into the trained UNet model in order to get the 22nd mask predictions.

```python
python3 file_path_to_unet_pred.py --dir file_path_to_hidden-practice2-sun5pm
```

For example:
```python
python3 unet_pred.py --dir hidden-practice2-sun5pm
```

## Conclusion

- The output_array.npy corresponds to the final numpy array we submitted to the final leaderboard. 

## Model Explanation

Here's an overview of each code's purpose and functionality:

1. **unet_model.py**: Defines the structure of the UNet model for 49 classes.
2. **unet_train.py**: Performs the training process for UNet using data in "Dataset_Student/train." Specify the epochs for training. Ensure correct paths for model, LOAD_MODEL, and train_root_dir.

   Example command: `python3 unet_train.py --epochs 50`

3. **unet_val.py**: Conducts the validation process for UNet using data from "Dataset_Student/val." Ensure accuracy by verifying the correct model path and val_dir. For comprehensive evaluation, leverage visualization functions from `unet_utils.py`. Confirm the correctness of the `plot_path` parameter for both `plot_and_save_learning_curves` and `plot_and_save_jaccard_idx` functions within `unet_utils.py`.

   Example command: `python3 unet_eval.py`

4. **unet_pred.py**: Performs prediction using the trained UNet model. The input is predictions from the PredNET model (2000 images with naming structure: `test_video{video_number}_9y_11.png`). After UNet prediction, it outputs the prediction masks as one np.array, named "output_array.npy."

   Example command: `python3 unet_pred.py --dir directory_of_the_prednet's_output_folder`


