from config import * 
from config import config
 



class model_1(nn.Module):
    def __init__(self,l0,l1=64,l2=32,l3=16,dropout_size=0.2):
        super().__init__()

        self.l1 = nn.Linear(l0,l1)
        self.bn1 = nn.BatchNorm1d(l1)
        self.dropout1 = nn.Dropout(dropout_size)

        self.l2 = nn.Linear(l1,l2)
        self.bn2 = nn.BatchNorm1d(l2)

        self.l3 = nn.Linear(l2,l3)
        self.bn3 = nn.BatchNorm1d(l3)


        self.l4 = nn.Linear(l3,1)

    def forward(self,x):

        x = self.l1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        x = self.l3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)

        x = self.l4(x)
        return x 



class flexible_model(nn.Module):
    def __init__(self,layer_config,input_size):
        super().__init__()

        layers = []
        previous_layer = input_size

        for layer in layer_config:

            activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "None": nn.Identity()
                            }




            layers.append(nn.Linear(previous_layer,layer["size"]))

            if layer.get("batch_norm",False) == True :
                layers.append(nn.BatchNorm1d(layer["size"]))


            layers.append(activation_map[layer["activation_function"]])

            if layer.get("dropout",0.0) > 0 :
                layers.append(nn.Dropout(layer.get("dropout")))


            previous_layer = layer["size"]

        self.layers = torch.nn.Sequential(*layers)

    def forward(self,x):
        return self.layers(x)






class flexible_model_sweep(nn.Module):
    def __init__(self,layer_config,input_size):
        super().__init__()






        layers = []
        previous_layer = input_size





        for layer in layer_config:

            activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "None": nn.Identity()
                            }




            layers.append(nn.Linear(previous_layer,layer["size"]))

            if layer.get("batch_norm",False) == True :
                layers.append(nn.BatchNorm1d(layer["size"]))


            layers.append(activation_map[layer["activation_function"]])

            if layer.get("dropout",0.0) > 0 :
                layers.append(nn.Dropout(layer.get("dropout")))


            previous_layer = layer["size"]

        self.layers = torch.nn.Sequential(*layers)


    
    def forward(self,x):
        return self.layers(x)




