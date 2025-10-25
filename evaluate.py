from config import *








def test(test_dl,model):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)



    mse_test,mae_test,r2_test,rmse_test = [],[],[],[]


    mse_test_org,mae_test_org,r2_test_org,rmse_test_org = [],[],[],[]






    
    y_true_test = 0
    y_pred_test = 0
    y_pred_test_org = 0


    with torch.no_grad():

        for x_test , y_test  in test_dl :

            x_test = x_test.to(device)
            y_test = y_test.to(device)



            model.eval()
                

            y_hat_test = model(x_test)
            
            y_true_test = (y_test.numpy().tolist())
            y_pred_test = (y_hat_test.detach().numpy().tolist())


            y_true_test_org= np.expm1(y_test.numpy())
            y_pred_test_org= np.expm1(y_hat_test.numpy())
        

    


        mse_test_epoch = mean_squared_error(y_true_test,y_pred_test)
        mse_test.append(mse_test_epoch)

        mae_test_epoch = mean_absolute_error(y_true_test,y_pred_test)
        mae_test.append(mae_test_epoch)

        r2_test_epoch =r2_score(y_true_test,y_pred_test)
        r2_test.append(r2_test_epoch)

        rmse_test_epoch = root_mean_squared_error(y_true_test,y_pred_test)
        rmse_test.append(rmse_test_epoch)




        # with original y range 


        mse_test_epoch_org = mean_squared_error(y_true_test_org,y_pred_test_org)
        mse_test_org.append(mse_test_epoch_org)

        mae_test_epoch_org = mean_absolute_error(y_true_test_org,y_pred_test_org)
        mae_test_org.append(mae_test_epoch_org)

        r2_test_epoch_org =r2_score(y_true_test_org,y_pred_test_org)
        r2_test_org.append(r2_test_epoch_org)

        rmse_test_epoch_org = root_mean_squared_error(y_true_test_org,y_pred_test_org)
        rmse_test_org.append(rmse_test_epoch_org)



        print("test")
        print("==============================")

        print("mse_test_epoch",mse_test_epoch)
        print("mae_test_epoch",mae_test_epoch)
        print("r2_test_epoch",r2_test_epoch)
        print("rmse_test_epoch",rmse_test_epoch)

        print("+++++++++")


        print("mse_test_epoch_org",mse_test_epoch_org)
        print("mae_test_epoch_org",mae_test_epoch_org)
        print("r2_test_epoch_org",r2_test_epoch_org)
        print("rmse_test_epoch_org",rmse_test_epoch_org)

        print("==============================")


        wandb.log({
                

            "mse_test_epoch":mse_test_epoch,
            "mae_test_epoch":mae_test_epoch,
            "r2_test_epoch":r2_test_epoch,
            "rmse_test_epoch":rmse_test_epoch,



            "mse_test_epoch_org":mse_test_epoch_org,
            "mae_test_epoch_org":mae_test_epoch_org,
            "r2_test_epoch_org":r2_test_epoch_org,
            "rmse_test_epoch_org":rmse_test_epoch_org,


        })



