import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "/Users/shaanil/Documents/Project/Ai-vision/Sign_language"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "yohanabhishek/ssl400-dynamic-sri-lankan-sign-language-dataset",file_path)


print("First 5 records:", df.head())