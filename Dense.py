class Dense:
    
    def __init__(self,X,y,layers,learning_rate,epochs):
        self.X=X
        self.y=y
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.layers=layers
        self.current_forward_result=None
        self.forward_pass_result=None
        self.netowrk=None
    
    def ReLU(self,leaky=False,negative_slope=None): # leaky is a flag for leaky relu, negative slope for leaky relu. 
        if leaky and negative_slope!=None:
            try:
                if negative_slope>0:
                    raise ValueError("Negative slope accepts integers only.")            
                return np.where(self.current_forward_result<0,self.current_forward_result*negative_slope,self.current_forward_result)
            
            except ValueError as ve:
                print(f"an unexpected error{ve}")
                
            except Exception as e:            
                print(f"An unexpected error {e}")
                raise 
        return np.maximum(0,self.current_forward_result)
    
    
    def gradient_ReLU(x):
        return np.where(x>0,1,0)
    
    def normal_initialization(fraction,weight_size):
        fract=np.sqrt(2/fraction)
        return np.random.normal(0,fract,size=weight_size)
    
    def uniform_initialization(fraction,weight_size):
        fract= np.sqrt(6/fraction)
        return np.random.uniform(-fract,fract,size=weight_size)

    
   
    def initialize_layer(layer_in,layer_out,mode="he",method="uniform",bias=True):
        """ Returns weights and biases initialized as response to input information

        Args:
            layer_in (_type_): _description_
            layer_out (_type_): _description_
            mode (str, optional): _description_. Defaults to "he".
            method (str, optional): _description_. Defaults to "uniform".

        Raises:
            ValueError: _description_
        """
        
        method_map={
            "uniform":uniform_initialization,
            "normal":normal_initialization
        }


        
        weights=np.zeros((layer_in,layer_out))
        if bias:
            biases=np.zeros((1,layer_out))
        
        
        try:        
            calcluation=0
            
            if mode.lower()=='random': return np.random.random(size=(layer_in,layer_out))
            
            elif mode.lower()=="he":
                calcluation=1/(np.sqrt(layer_in+layer_out))
                
            elif mode.lower()=="xavier":
                calcluation=1/(np.sqrt(layer_in))
                
            else:
                raise ValueError("Only accepts 'random','he' and 'xavier' string as arguments.")

            weights=method_map[method](calcluation,(layer_in,layer_out))
            
            
        
        except Exception as e:
            print(f"Error occured: {e}")    

        
        return {"weight":weights,"bias":biases}
    
    def forward_pass(input_network,X):
        network=input_network.copy()
        result=X @ network['Layer0']["weight"] + network['Layer0']["bias"]
        network.pop('Layer0')
        
        for layer_name in network:
            params = network[layer_name]
            W = params["weight"]
            b = params["bias"]
            result=result @ W +b
            
        return result 
        