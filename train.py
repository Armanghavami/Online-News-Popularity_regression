
from config import *
from config import config 





def train(model,epochs,train_dl,valid_dl,loss_function,optimizer,scheduler,early_stopping_delta,early_stopping_patience,l1_lambda,saved_model_sweep=False,saved_model=False,continue_training=False):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    mse_train = []
    mae_train = []
    rmse_train = []
    r2_train = []


    mse_valid = []
    mae_valid = []
    rmse_valid = []
    r2_valid = []

    best_val_loss = float('inf')


    if continue_training == True :
        model.load_state_dict(torch.load("/Users/arman/Documents/machin learning/Model Architectures/deep feed forward_nn /Online News Popularity_regression/saved_w.pth"))

        print("✅ Loaded previous weights")


    for epoch in range(epochs):
        model.train()
        batch_samples_train = 0
        batch_loss_train = 0 
        y_true_train = []
        y_pred_train = []


        for x_train , y_train in train_dl:
            x_train = x_train.to(device)
            y_train = y_train.to(device)


            optimizer.zero_grad()

            y_hat_train = model(x_train)
            loss = loss_function(y_hat_train,y_train)


            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss_train = loss + l1_lambda * l1_norm


            batch_samples_train += x_train.size(0)
            batch_loss_train += loss * x_train.size(0)

            loss_train.backward()
            optimizer.step()
                       




            y_true_train.extend(y_train.numpy().tolist())        
            y_pred_train.extend(y_hat_train.detach().numpy().tolist())
            


        mse_train_epoch = (batch_loss_train/batch_samples_train).item()
        mse_train.append(mse_train_epoch)


        mae_train_epoch = mean_absolute_error(y_true_train,y_pred_train)
        mae_train.append(mae_train_epoch)


        r2_train_epoch =r2_score(y_true_train,y_pred_train)
        r2_train.append(r2_train_epoch)


        rmse_train_epoch = root_mean_squared_error(y_true_train,y_pred_train)
        rmse_train.append(rmse_train_epoch)



        # the valid set 

        batch_samples_valid = 0 
        batch_loss_valid = 0 

        y_true_valid = []
        y_pred_valid = []




        for x_valid , y_valid in valid_dl:

            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)


            model.eval()
            y_hat_valid = model(x_valid)
            loss_valid = loss_function(y_hat_valid,y_valid)

            batch_samples_valid += x_valid.size(0)
            batch_loss_valid += loss_valid * x_valid.size(0)


            y_true_valid.extend(y_valid.numpy().tolist())        
            y_pred_valid.extend(y_hat_valid.detach().numpy().tolist())
            

        
        





        mse_valid_epoch = (batch_loss_valid/batch_samples_valid).item()
        mse_valid.append(mse_valid_epoch)


        mae_valid_epoch = mean_absolute_error(y_true_valid,y_pred_valid)
        mae_valid.append(mae_valid_epoch)


        r2_valid_epoch =r2_score(y_true_valid,y_pred_valid)
        r2_valid.append(r2_valid_epoch)


        rmse_valid_epoch = root_mean_squared_error(y_true_valid,y_pred_valid)
        rmse_valid.append(rmse_valid_epoch)


        scheduler.step(mse_valid_epoch)







        

        if mse_valid_epoch < best_val_loss - early_stopping_delta:
            best_val_loss = mse_valid_epoch
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break




        print("________________________________")
        print("epoch:",epoch)
        print("+++++++++")
        print("train")
        print("mse_train",mse_train_epoch)
        print("mae_train",mae_train_epoch)
        print("r2_train",r2_train_epoch)
        print("rmse_train",rmse_train_epoch)
        print("+++++++++")
        print("valid")
        print("mse_valid",mse_valid_epoch)
        print("mae_valid",mae_valid_epoch)
        print("r2_valid",r2_valid_epoch)
        print("rmse_valid",rmse_valid_epoch)





        wandb.log({
        "mse_train":mse_train_epoch,
        "mae_train":mae_train_epoch,
        "r2_train":r2_train_epoch,
        "rmse_train":rmse_train_epoch,


        "mse_valid":mse_valid_epoch,
        "mae_valid":mae_valid_epoch,
        "r2_valid":r2_valid_epoch,
        "rmse_valid":rmse_valid_epoch,

        
        "Lr":optimizer.param_groups[0]['lr'],

        
        })
    if saved_model == True :
    
        torch.save(model.state_dict(),"saved_w.pth")
        print("✅ Saved final model")

    if saved_model_sweep == True :
    
        torch.save(model.state_dict(),"saved_sweep_w.pth")
        print("✅ Saved final model")




    return mse_train,mae_train,r2_train,rmse_train,mse_valid,mae_valid,r2_valid,rmse_valid









 