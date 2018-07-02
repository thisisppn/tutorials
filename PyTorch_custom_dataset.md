
### Creating custom Dataset classes in PyTorch

PyTorch, like Tensorflow is a Deep Learning Framework for Python, that is especially popular in the research space. Unlike Tensorflow which is better when it comes to deploying neural network to production, PyTorch is simple to setup and easy to debug and is more pythonic. Even with it’s simplicity, it offers maximum flexibility and speed.

Along with the research field, PyTorch is also quite popular among the community of people are into kaggle contests and similar online contests like them. I’ll try to give some insight on mainly the Image related contests there. All Deep Learning frameworks have very easy support for Datasets where each class of images are in it’s individual folders.

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png
    ...
    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png

Unfortunately though, most of the contest online don’t have their data in the above format. They usually have all the images inside one folder called “train” and then there is a CSV which maps the filename with it’s respective category. There are two ways to handle this — 

-   Write a script to iterate through the CSV and create the folders for all the available classes and move the images to the respective folders
-   Create custom Dataset classes to handle it.

Which one is a better choice, would depend on your situation, but I am here to give an insight of how to do the second point.

**DATASET** — I will be using a data set which is currently being used in a [Deep Learning Beginner](https://www.hackerearth.com/challenge/competitive/deep-learning-beginner-challenge/machine-learning/predict-the-energy-used-612632a9-9de79188/) challenge at Hackerearth. If you want to have a look at it, you can download it from [this](https://s3-ap-southeast-1.amazonaws.com/he-public-data/DL%23+Beginner.zip) link provided in the contest page. Note that you need to use 7zip to unzip the folder.

The dataset has two folders train and test, which has all the images in it. The labelling is done in a CSV file which looks something like this. 

|  | Image_id | Animal |
|--|--|--|
|0| Img-1.jpg | hippopotamus |
|1| Img-2.jpg | squirrel |
|2| Img-3.jpg | grizzly+bear |
|3| Img-4.jpg | ox |
|4| Img-5.jpg | german+shepherd |

Let's visualise some of the data. 

![Some images from the data set](output_7_0.png)


Since neural networks are nothing but mathematical operations happening in the background, they won’t support the classnames which are strings. Here you could apply label encoding to convert the class names into integers, but I wanted to simulate the ImageFolder’s `class_to_idx` feature which converts the classes into integers, and stores it in the object.

```python
# Simulate the class_to_idx of the ImageFolder function of PyTorch  
def find_classes(dataframe):  
    classes = list(set(dataframe.iloc[:, 1].values)) # Get unique class names  
    classes.sort()  
    class_to_idx = {classes[i]: i for i in range(len(classes))}  
    return classes, class_to_idx
```
For the custom dataset class, we inherit the dataset class of PyTorch
```python
class AnimalDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.csv_mapping = pd.read_csv(csv_file)
        self.root_dir = data_dir
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(self.csv_mapping)
        
    def __len__(self):
        return len(self.csv_mapping)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_mapping.iloc[idx, 0])
        image = Image.open(img_name) # For the transformations to work, we need to use 
        image = image.convert("RGB") # Some PNG images come up with 4 channels, which cause errors. 
        label = self.class_to_idx[self.csv_mapping.iloc[idx, 1]]
        
        if self.transform:
            image = self.transform(image)

        return (image, label)
    
    def get_class_to_idx(self):
        return self.class_to_idx
```

