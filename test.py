from crackseg.utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



if __name__ == "__main__":
    image_path = 'data/CrackLS315_IMAGE'
    mask_path = 'data/CrackLS315_MASK'
    image_type = 'jpg'
    train_data = CustomDataset(image_path , mask_path,image_type,is_resize=True)
    train_DataLoader = DataLoader(train_data , batch_size=1)
    
    
    for i in tqdm(train_DataLoader,total=len(train_DataLoader)):
        batch = i
        print(batch[0].shape)
        print(batch[1].shape)
        
        
    
    
    
    