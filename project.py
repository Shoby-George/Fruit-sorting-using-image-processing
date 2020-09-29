import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure


def Loadimages(n,fruit):
       
    mypath="C:/Users/LALU/Desktop/Fruits/"+str(fruit)+"/"
    file = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
    for i in range(n):
       s=str(mypath)+str(file[i])      
       img = cv2.imread(s)
       image = cv2.resize(img,(100,100))
       Images.append(image)
       Fruits.append(str(fruit))

def Feature1(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
   ftr,_=hog(gray, orientations=15, pixels_per_cell=(10, 10),cells_per_block=(1, 1), visualize=True, multichannel=False) 
  # print(ftr)
   return ftr

def Feature2(image):
    features = []
    imag= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV = cv2.split(imag)
    N=(360,1,1)
    for (hsv,n) in zip(HSV,N):
        hist = cv2.calcHist([hsv], [0], None, [250], [0, n])
        features.extend(hist)   
    return np.array(features).flatten()   

def FeatureTest(image):
    features = []
    imag= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    HSV = cv2.split(imag)
    colors = ("r", "g", "b")
    N=(360,1,1)
    LABEL=("HUE","SATURATION","VALUE")
    plt.figure()    
    
    for (hsv, col,n,label) in zip(HSV, colors,N,LABEL):
        hist = cv2.calcHist([hsv], [0], None, [250], [0, n])
        plt.title(label)
        plt.xlabel("Bins")
        plt.ylabel("Number of Pixels")
        plt.plot(hist, color = col)    
        plt.xlim([0, 250]) 
        plt.ylim([0, 1000])
        features.extend(hist)   
        plt.show()
    return np.array(features).flatten()

   
Images=[]
Fruits=[]
Features=[]
List=("Banana","Lemon","Lime","Orange","Peach","Pear","Red Apple","Green Apple")
for i in range(len(List)):
    Loadimages(50,List[i])
   
  
#Feature extraction
for i in range(len(Images)):
    cpy=np.copy(Images[i])
    ftr1=Feature1(cpy)
    ftr2=Feature2(Images[i])
    Feat=np.hstack((ftr1,ftr2))
    Features.append(Feat)

#print(Features[0].shape)    
Features=np.asarray(Features)    
X_train, X_test, Y_train, Y_test = train_test_split(Features,Fruits, test_size = 10/100, random_state = 30)           



lda = LDA(n_components =5)
lda.fit_transform(X_train,Y_train)
lda.transform(X_test)
prdt=lda.predict(X_test)
#print(prdt)

var= lda.explained_variance_ratio_
y= np.arange(1,len(var)+1)
plt.bar(y,var, align='center',width=0.5,Color='green', alpha=0.9)
plt.title("Variance ratio ")
plt.xlabel('Dimensions')
plt.ylabel('Variance ratio')
plt.show()

print("Accuracy  for LDA is ",accuracy_score(Y_test, prdt)*100,'%')


#Testing
timg = plt.imread("C:/Users/LALU/Desktop/Fruits/Test/2.png")
timage=np.copy(timg)
timage = cv2.resize(timg,(100,100))
ftr1=Feature1(timage)
ftr2=FeatureTest(timage)
t=np.hstack((ftr1,ftr2))

predict=lda.predict(t.reshape(1, -1))
plt.subplot(1,2,1)
plt.imshow(timg)
plt.title(predict)
plt.xticks([])
plt.yticks([])


gray = cv2.cvtColor(timage ,cv2.COLOR_BGR2GRAY)
ftr,hog_image=hog(gray, orientations=30, pixels_per_cell=(10, 10),cells_per_block=(1, 1), visualize=True, multichannel=False) 
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))
plt.subplot(1,2,2)
plt.imshow(hog_image_rescaled) 
plt.title("HOG Image")
plt.xticks([])
plt.yticks([])
plt.show()
