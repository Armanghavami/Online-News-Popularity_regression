from config import *
from config import pd , train_test_split , StandardScaler , torch , TensorDataset , DataLoader , MinMaxScaler
from config import *
from data import data_load, data_normalization,load_dataset
#from main import train_dl,valid_dl,loss_function,optimizer,scheduler

import random
import numpy as np
from model import model_1 ,flexible_model,flexible_model_sweep
from train import train
from evaluate import test 



config_1 = {
    'method': 'bayes',
    'metric': {'name': 'mse_valid', 'goal': 'minimize'},



    "parameters": {

        "l1_lambda": {
            "distribution": "log_uniform_values",
            "min": 5e-4,
            "max": 1e-2
        },
        "weight_decay": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-3
                },

        
        "l1_size":
            { "values": [16, 32, 64, 128] },
        "l1_batch_norm":
            { "value": True },
        "l1_dropout":
            { "values": [0.2, 0.4, 0.6] },
        "l1_activation_function":
            { "value": "leaky_relu" },

        "l2_size":
            { "values": [16, 32, 64, 128] },
        "l2_batch_norm":
            { "value": True },
        "l2_dropout":
            { "values": [0.2, 0.4, 0.6] },
        "l2_activation_function":
            { "value": "leaky_relu" },

        "l3_size":
            { "values": [16, 32, 64, 128] },
        "l3_batch_norm":
            { "value": True },
        "l3_dropout":
            { "value": 0 },
        "l3_activation_function":
            { "value": "leaky_relu" },

        "l4_size":
            { "values": [16, 32, 64, 128] },
        "l4_batch_norm":
            { "value": True },
        "l4_dropout":
            { "value": 0 },
        "l4_activation_function":
            { "value": "leaky_relu" },

        "l5_size":
            { "value": 1 },
        "l5_batch_norm":
            { "value": False },
        "l5_dropout":
            { "value": 0 },
        "l5_activation_function":
            { "value": "None" }
        



    },

}

"""
{"size":[16,32,64,128],"batch_norm":True,"dropout":[0.2,0.4,0.6],"activation_function":"leaky_relu"},
{"size":[16,32,64,128],"batch_norm":True,"dropout":[0.2,0.4,0.6],"activation_function":"leaky_relu"},
{"size":[16,32,64,128],"batch_norm":True,"dropout":0,"activation_function":"leaky_relu"},
{"size":[16,32,64,128],"batch_norm":True,"dropout":0,"activation_function":"leaky_relu"},
{"size":1,"batch_norm":False,"dropout":0,"activation_function":"None"},]
"""




def sweep_main():

    wandb.init( config=config_1)
    config = wandb.config
    
    config.saved_model_sweep =True 
    config.continue_training =False


    config.data_path ="/Users/arman/Documents/machin learning/Model Architectures/deep feed forward_nn /Online News Popularity_regression/data/OnlineNewsPopularity.csv"
    config.batch_size =128
    config.data_normalization_method ="z_score"
    config.train_size = 0.5
    config.epochs = 1
    config.learning_rate=0.05
    config.samples_count = 3950
    config.lr_factor=0.75
    config.lr_patioence=3
    config.early_stopping_delta=0
    config.early_stopping_patience=10




    layer_config = []
    for i in range(1, 6):
        layer = {
            "size": getattr(config, f"l{i}_size"),
            "batch_norm": getattr(config, f"l{i}_batch_norm"),
            "dropout": getattr(config, f"l{i}_dropout"),
            "activation_function": getattr(config, f"l{i}_activation_function"),
        }
        layer_config.append(layer)





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

    #model = flexible_model_sweep(config.get("layer_config"),future_size)
    model = flexible_model_sweep(layer_config,future_size)


    wandb.watch(model, log="all", log_freq=10)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"],weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["lr_factor"], patience=config["lr_patioence"])


    (mse_train,mae_train,r2_train,rmse_train,
    mse_valid,mae_valid,r2_valid,rmse_valid) = train(model,config["epochs"],train_dl,valid_dl,loss_function,
                                                    optimizer,scheduler,config["early_stopping_delta"],config["early_stopping_patience"],config["l1_lambda"]
                                                    ,config["saved_model_sweep"])

    # mse last item for the sweep 
    #mse_valid_lastitem = mse_valid[-1]
    # for the tests

    #(mse_test,mae_test,r2_test,rmse_test,
    #mse_test_org,mae_test_org,r2_epoch_org,rmse_test_org) = 
    test(test_dl,model)


    '''''
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
    '''''





    wandb.config.update(config)



    wandb.finish()










if __name__ == "__main__":
    sweep_id = wandb.sweep(config_1, project="news_sweep")
    wandb.agent(sweep_id, function=sweep_main,count=1)

