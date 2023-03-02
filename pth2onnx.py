import torch.onnx 
import os
import torch


from models.model_MobileNetV2 import MobileNetV2


def GetAllFilePath(oneDirectoryPath):

    primesense_list = []
    oneDirectory = os.walk(oneDirectoryPath)
    for path, dir_list, file_list in oneDirectory:
        for file_name in file_list:
            str_path = os.path.join(path, file_name)
            if "pth" in file_name:
                primesense_list.append(str_path)

    return primesense_list

onnx_path = "D:\\MyNetworks\\datasets\\lemon\\save\\vseemodel.onnx"

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(5, 3, 160, 160, requires_grad=True)  
    # dummy_input = torch.randn(1, 3, 320, 320, requires_grad=True)     # 搭配dynamic_axes可以动态调整
    
    torch.onnx.export(model,                                                        # model being run 
         dummy_input,                                                               # model input (or a tuple for multiple inputs) 
         onnx_path,                                                                 # where to save the model  
         export_params=True,                                                        # store the trained parameter weights inside the model file 
         opset_version=10,                                                          # the ONNX version to export the model to 
         do_constant_folding=True,                                                  # whether to execute constant folding for optimization 
         input_names = ['input1'],                                                  # the model's input names 
         output_names = ['output1'],                                               # the model's output names 
         ) 
    
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 

    pthModel_path = GetAllFilePath(r"D:\\MyNetworks\\datasets\\chestnut\\save\\temp")
    
    # Let's load the model we just created and test the accuracy per label 
    # for i in range(len(pthModel_path)):
    for path in pthModel_path:
        model = MobileNetV2(8) 
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(path)) 
        onnx_path = path.replace("pth", "onnx")
        # Conversion to ONNX 
        Convert_ONNX()










# onnx_path = "D:\\MyNetworks\\datasets\\lemon\\save\\vseemodel.onnx"

# #Function to Convert to ONNX 
# def Convert_ONNX(): 

#     # set the model to inference mode 
#     model.eval() 

#     # Let's create a dummy input tensor  
#     dummy_input = torch.randn(5, 3, 160, 160, requires_grad=True)  
#     # dummy_input = torch.randn(1, 3, 320, 320, requires_grad=True)     # 搭配dynamic_axes可以动态调整

#     # # Export the model   
#     # torch.onnx.export(model,                                                        # model being run 
#     #      dummy_input,                                                               # model input (or a tuple for multiple inputs) 
#     #      onnx_path,                                                                 # where to save the model  
#     #      export_params=True,                                                        # store the trained parameter weights inside the model file 
#     #      opset_version=10,                                                          # the ONNX version to export the model to 
#     #      do_constant_folding=True,                                                  # whether to execute constant folding for optimization 
#     #      input_names = ['input1'],                                                  # the model's input names 
#     #      output_names = ['output1'],                                               # the model's output names 
#     #      dynamic_axes={'input1' : {0 : 'batch_size'},                               # variable length axes 
#     #                             'output1' : {0 : 'batch_size'}}) 
    
#     torch.onnx.export(model,                                                        # model being run 
#          dummy_input,                                                               # model input (or a tuple for multiple inputs) 
#          onnx_path,                                                                 # where to save the model  
#          export_params=True,                                                        # store the trained parameter weights inside the model file 
#          opset_version=10,                                                          # the ONNX version to export the model to 
#          do_constant_folding=True,                                                  # whether to execute constant folding for optimization 
#          input_names = ['input1'],                                                  # the model's input names 
#          output_names = ['output1'],                                               # the model's output names 
#          ) 
    
#     print(" ") 
#     print('Model has been converted to ONNX')


# if __name__ == "__main__": 

#     # Let's build our model 
#     #train(5) 
#     #print('Finished Training') 

#     # Test which classes performed well 
#     #testAccuracy() 

#     # Let's load the model we just created and test the accuracy per label 
#     model = MobileNetV2(8) 
#     path = "D:\\MyNetworks\\datasets\\lemon\\save\\lemon_MobileNetV2.pth" 
#     model.load_state_dict(torch.load(path)) 

#     # Test with batch of images 
#     #testBatch() 
#     # Test how the classes performed 
#     #testClassess() 
 
#     # Conversion to ONNX 
#     Convert_ONNX()