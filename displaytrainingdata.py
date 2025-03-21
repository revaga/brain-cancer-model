# displaying data

# importing libraries
import matplotlib.pyplot as plt
import random
from PIL import Image 
import torchvision.transforms as transforms


training_glioma = []
for i in range(10, 1321): #skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    training_glioma.append("Training/glioma/Tr-gl_" + val+ ".jpg")

    
training_meningioma = []
for i in range(10, 1339): #1338 images, skipped first 10
    val = str(i)
    while(len(val) !=4):
        val = "0"+val
    training_meningioma.append("Training/meningioma/Tr-me_" + val+ ".jpg")

    
training_notumor = []
for i in range(10,1595): #1594 images, first 10 ommitted
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    training_notumor.append("Training/notumor/Tr-no_" + val+ ".jpg")
    #print("Training/notumor/Tr-no_" + val+ ".jpg")


training_pituitary = []
for i in range(10,1457): #1456 images, first 10 ommitted
    val = str(i)
    while(len(val) != 4):
        val = "0" + val
    training_pituitary.append("Training/pituitary/Tr-pi_" + val+ ".jpg")
    #print(training_pituitary[i])
    
"""
#showing images

for i in range(0,2):
    img = Image.open(training_glioma[i])
    img.show()
"""

# Convert training images into tensors 
# ref: https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((256,256))
])

traintensor_glioma = []
for i in range(0, len(training_glioma)):
    img = Image.open(training_glioma[i]).convert('L')
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_glioma.append(img_tensor)

traintensor_meningioma = []
for i in range(0, len(training_meningioma)):
    img = Image.open(training_meningioma[i]).convert('L')
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_meningioma.append(img_tensor)

traintensor_notumor = []
for i in range(0, len(training_notumor)):
    img = Image.open(training_notumor[i]).convert('L')
    img_tensor = transform(img)
    traintensor_notumor.append(img_tensor)

traintensor_pituitary = []
for i in range(0, len(training_pituitary)):
    img = Image.open(training_pituitary[i]).convert('L')
    img_tensor = transform(img)
    #print(img_tensor)
    traintensor_pituitary.append(img_tensor)



fig, axs = plt.subplots(4,4)
plt.title("Glioma")
k=0
for i in range(0, 4):
    for j in range(0,4):
        img = Image.open(training_glioma[k]).convert('L')
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        k = k + 1

plt.tight_layout()
plt.savefig("glioma.png")
plt.close()


fig, axs = plt.subplots(4,4)
plt.title("Meningioma")
k=0
for i in range(0, 4):
    for j in range(0,4):
        img = Image.open(training_meningioma[k]).convert('L')
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        k = k + 1
plt.tight_layout()
plt.savefig("meningioma.png")
plt.close()


fig, axs = plt.subplots(4,4)
plt.title("No Tumor")
k=0
for i in range(0, 4):
    for j in range(0,4):
        img = Image.open(training_notumor[k]).convert('L')
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        k = k + 1

plt.tight_layout()
plt.savefig("notumor.png")
plt.close()


fig, axs = plt.subplots(4,4)
plt.title("Pituitary")
k=0
for i in range(0, 4):
    for j in range(0,4):
        img = Image.open(training_pituitary[k]).convert('L')
        axs[i,j].imshow(img)
        axs[i,j].axis('off')
        k = k + 1

plt.tight_layout()
plt.savefig("pituitary.png")
plt.close()

for imgnum in range(0, 6):
    fig, axs = plt.subplots(4,4)
    for i in range(0, 4):
        k=random.randint(0,1000)
        for j in range(0,4):
            if(i == 0):
                img = Image.open(training_glioma[k]).convert('L')
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                strk = "glioma: " + str(k)
                axs[i,j].set_title(strk)
            
            elif i == 1:
                img = Image.open(training_meningioma[k]).convert('L')
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                strk = "meningioma: " + str(k)
                axs[i,j].set_title(strk)
        
            elif i == 2:
                img = Image.open(training_notumor[k]).convert('L')
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                strk = "notumor: " + str(k)
                axs[i,j].set_title(strk)


            elif i == 3:
                img = Image.open(training_pituitary[k]).convert('L')
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                strk = "pituitary: " + str(k)
                axs[i,j].set_title(strk)

            k = k + 1

    plt.tight_layout()
    strsave = "all" + str(imgnum) + ".png"
    plt.savefig(strsave)
    plt.close()
