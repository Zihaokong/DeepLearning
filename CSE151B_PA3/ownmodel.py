import torch.nn as nn



# Defining your CNN model
# We have defined the baseline model


class baseline_Net(nn.Module):

    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        self.b1 = nn.Sequential(
            # in channel 3, out channel 64, kernel size 3
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            # in channel 64, out channel 128, size 3
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            # in channel 128, out channel 128
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((3,3)),
            nn.Dropout(),
            
            
            nn.Conv2d(256, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 512)
        out4 = self.fc2(self.fc1(out_flat))

        return out4


# class baseline_Net(nn.Module):

#     def __init__(self, classes):
#         super(baseline_Net, self).__init__()
#         self.b1 = nn.Sequential(
#             # in channel 3, out channel 64, kernel size 3
#             nn.Conv2d(3, 96, 11,stride=4),
#             nn.ReLU(inplace=True),
            
            
#             nn.MaxPool2d((3,3),stride=2),
            
            
#             nn.Conv2d(96, 256, 5,stride=2,padding=2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
            
            
#             nn.MaxPool2d((3,3),stride=2),
#         )
#         self.b2 = nn.Sequential(
#             # in channel 64, out channel 128, size 3
#             nn.Conv2d(256, 384, 3,padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(384),
#             nn.Dropout(),           
#             nn.Conv2d(384, 384, 3,padding=1),
#             nn.ReLU(inplace=True),   
#             nn.BatchNorm2d(384),
#             nn.Dropout(),
#             nn.Conv2d(384, 384, 3,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Conv2d(384, 384, 3,padding=1),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
# #             nn.Conv2d(384, 384, 3,padding=2),
# #             nn.ReLU(inplace=True),  
                        
#             nn.BatchNorm2d(384),
#             nn.MaxPool2d((3,3),stride=2)
#         )
#         self.b3 = nn.Sequential(
            
# #             nn.Conv2d(384, 456, 3, stride=1),
# #             nn.ReLU(inplace=True),
# #             nn.BatchNorm2d(456),
# #             nn.MaxPool2d((3,3),stride=2),
#             nn.Flatten(),
            

#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(384*4, 1024),
#             nn.Dropout(),
#             nn.ReLU(inplace=True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, classes)
#         )

#     def forward(self, x):
#         out1 = self.b3(self.b2(self.b1(x)))
        
#         #print(out1.shape)
        
        
#         out4 = self.fc2(self.fc1(out1))

#         return out4


