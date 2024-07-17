from crackseg.utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



if __name__ == "__main__":
    out_list = [["hello","world"]]
    str1 = "dsban"
    str2 = "kello"
    k = zip(str1,str2)
    print(list(k))
    
    for output,target in out_list:
        print(output)
        print(target)
    
    
    
    