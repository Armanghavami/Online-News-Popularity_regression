from config import *
from config import config


def data_load(data_path,train_size=0.5,samples_count=5000):
    data_set = pd.read_csv(data_path)

    # droping the non-predictive col 
    data_set = data_set.drop(columns=["url"," timedelta"])

    binary_col = []
    non_binary_col = []

    for col in data_set.columns:

        unique_value = data_set[col].dropna().unique()
        if len(unique_value) == 2 and set(unique_value).issubset({0,1}):
            binary_col.append(col) 


    for col in data_set.columns:
        if col not in  binary_col and not col == " shares":
            non_binary_col.append(col)



    # picking n sample for the initial training for the limitation of computing power
    data_set_sample = data_set.sample(n=samples_count,random_state=42)

    x_sample = data_set_sample.drop(columns=" shares")
    y_sample = data_set_sample[" shares"]



    # train , test , valid data set 
    x_train,x_temp,y_train,y_temp = train_test_split(x_sample,y_sample,train_size=train_size,random_state=42)
    x_test,x_valid,y_test,y_valid = train_test_split(x_temp,y_temp,train_size=0.5,random_state=42)

    return x_train,x_test,x_valid,y_train,y_test,y_valid,non_binary_col,binary_col




#normalization of the data and making them into tensor  
def data_normalization(data_normalization_method ,x_train,x_test,x_valid,y_train,y_test,y_valid,non_binary_col,binary_col):


        # z_score normalization of the non binary data colums 

    if data_normalization_method == "z_score":
        z_score = StandardScaler()

        x_train_norm = z_score.fit_transform(x_train[non_binary_col])
        x_test_norm = z_score.transform(x_test[non_binary_col])
        x_valid_norm = z_score.transform(x_valid[non_binary_col])

    elif data_normalization_method == "min_max":

        min_max = MinMaxScaler()

        x_train_norm = min_max.fit_transform(x_train[non_binary_col])
        x_test_norm = min_max.transform(x_test[non_binary_col])
        x_valid_norm = min_max.transform(x_valid[non_binary_col])

    else :
        x_train_norm = x_train
        x_test_norm = x_test
        x_valid_norm = x_valid

    # adding back the binary col 


    x_train_binary = x_train[binary_col].values
    x_test_binary = x_test[binary_col].values
    x_valid_binary = x_valid[binary_col].values

    x_train_norm = np.hstack((x_train_norm, x_train_binary))
    x_test_norm = np.hstack((x_test_norm, x_test_binary))
    x_valid_norm = np.hstack((x_valid_norm, x_valid_binary))


    # normlization of the y value

    y_train_norm = np.log1p(y_train.values).reshape(-1, 1)
    y_test_norm = np.log1p(y_test.values).reshape(-1, 1)
    y_valid_norm = np.log1p(y_valid.values).reshape(-1, 1)




    # making the data into tensors 
    x_train_norm_tensor = torch.tensor(x_train_norm,dtype=torch.float32)
    x_test_norm_tensor = torch.tensor(x_test_norm,dtype=torch.float32)
    x_valid_norm_tensor = torch.tensor(x_valid_norm,dtype=torch.float32)

    y_train_norm_tensor = torch.tensor(y_train_norm,dtype=torch.float32)
    y_test_norm_tensor = torch.tensor(y_test_norm,dtype=torch.float32)
    y_valid_norm_tensor = torch.tensor(y_valid_norm,dtype=torch.float32)




    return x_train_norm_tensor,x_test_norm_tensor,x_valid_norm_tensor,y_train_norm_tensor,y_test_norm_tensor,y_valid_norm_tensor



def load_dataset(x_train_norm_tensor,x_test_norm_tensor,x_valid_norm_tensor,y_train_norm_tensor,y_test_norm_tensor,y_valid_norm_tensor,batch_size):

    # making a tensor dataset 
    train_ds = TensorDataset(x_train_norm_tensor,y_train_norm_tensor)
    valid_ds = TensorDataset(x_valid_norm_tensor,y_valid_norm_tensor)
    test_ds = TensorDataset(x_test_norm_tensor,y_test_norm_tensor)


    # loading the dataset 
    train_dl = DataLoader(train_ds,batch_size=batch_size)
    valid_dl = DataLoader(valid_ds,batch_size=batch_size)
    test_dl = DataLoader(test_ds,batch_size=64)

    future_size = x_train_norm_tensor.size(1)



    return train_dl,valid_dl,test_dl,future_size

 