from config import pd , train_test_split , StandardScaler , torch , TensorDataset , DataLoader , MinMaxScaler
from config import *
from data import data_load, data_normalization,load_dataset
#from main import train_dl,valid_dl,loss_function,optimizer,scheduler

from config import config
import random
import numpy as np
from model import model_1 ,flexible_model
from train import train
from evaluate import test 

wandb.init(project="news_regression",name="experiment_3.2",save_code=True,notes="",id="1")#mode= "disabled",resume="must"



# seed 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)   



# data manipulation and loading  
(x_train,x_test,x_valid,y_train,y_test
,y_valid,non_binary_col,binary_col) = data_load(config["data_path"],config["train_size"],config["samples_count"])



(x_train_norm_tensor,x_test_norm_tensor,x_valid_norm_tensor,
 y_train_norm_tensor,y_test_norm_tensor,y_valid_norm_tensor,) = data_normalization(config["data_normalization_method"],x_train,x_test,
                                                                   x_valid,y_train,y_test,y_valid,non_binary_col,binary_col)




train_dl,valid_dl,test_dl,future_size = load_dataset(x_train_norm_tensor,x_test_norm_tensor,x_valid_norm_tensor
                                                    ,y_train_norm_tensor,y_test_norm_tensor,y_valid_norm_tensor,config["batch_size"])
 


# model 
# fixed model 
#model = model_1(l0=future_size,)

model = flexible_model(config.get("layer_config"),future_size)


wandb.watch(model, log="all", log_freq=10)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["lr_factor"], patience=config["lr_patioence"])


(mse_train,mae_train,r2_train,rmse_train,
 mse_valid,mae_valid,r2_valid,rmse_valid) = train(model,config["epochs"],train_dl,valid_dl,loss_function,
                                                  optimizer,scheduler,config["early_stopping_delta"],config["early_stopping_patience"],config["l1_lambda"],config["saved_model"],config["continue_training"])


# for the tests

#(mse_test,mae_test,r2_test,rmse_test,
#mse_test_org,mae_test_org,r2_epoch_org,rmse_test_org) = 
test(test_dl,model)



# for shap Xai
x_test_np_shap = x_test_norm_tensor.numpy()

background_data = x_test_np_shap[:500]
samples_to_explain = x_test_np_shap[100:500]


model.eval()
explainer = shap.DeepExplainer(model, torch.tensor(background_data, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(samples_to_explain, dtype=torch.float32))

feature_names = non_binary_col + binary_col


shap_values_exp = shap.Explanation(shap_values[0].squeeze(), 
                                  base_values=explainer.expected_value.item(),
                                  data=samples_to_explain[0], 
                                  feature_names=feature_names)


shap.plots.waterfall(shap_values_exp, show=True,)















wandb.config.update(config)



wandb.finish()